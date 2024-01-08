bl_info = {
    "name": "Apply Shape Key to Basis",
    "author": "Mysteryem",
    "version": (0, 0, 2),
    "blender": (3, 3, 0),
    "location": "Properties > Data > Shape Key Specials menu > Apply Shape Key to Basis",
    "description": "Adds a tool for applying the active shape key to the Basis and propagating the change to dependent"
                   " shape keys.",
    "doc_url": "https://github.com/Mysteryem/blender-apply-shape-key-to-basis",
    "tracker_url": "https://github.com/Mysteryem/blender-apply-shape-key-to-basis/issues",
    "category": "Mesh",
}
#     Apply Shape Key to Basis Blender add-on
#     Copyright (C) 2022-2024 Thomas Barlow (Mysteryem)
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.

import bpy
from bpy.types import (
    Context,
    Key,
    Mesh,
    Object,
    Operator,
    ShapeKey,
)

from collections.abc import Iterator
from typing import cast, TypeVar

import numpy as np


# Utilities


T = TypeVar("T", bound=Operator)


def set_operator_description_from_doc(cls: type[T]) -> type[T]:
    """
    Clean up an Operator's .__doc__ for use as bl_description.
    :param cls: Operator subclass.
    :return: The Operator subclass argument.
    """
    doc = cls.__doc__
    if not doc or getattr(cls, "bl_description", None) is not None:
        # There is nothing to do if there is no `__doc__` or if `bl_description` has been set manually.
        return cls
    # Strip the entire string first to remove leading/trailing newlines and other whitespace.
    doc = doc.strip()
    if doc[-1] == ".":
        # Remove any trailing "." because Blender adds one automatically.
        doc = doc[:-1]
    # Remove leading/trailing whitespace from each line.
    cls.bl_description = "\n".join(line.strip() for line in doc.splitlines())
    return cls


class OperatorBase(Operator):
    @classmethod
    def poll_fail(cls, message: str):
        """
        Small helper to set the poll message and then return False.
        """
        cls.poll_message_set(message)
        return False


# Blender Classes


# As of Blender 4.0 (in commit d32748cdf4), modifying Shape Keys in Edit mode propagates changes to all Shape Keys
# recursively relative to the current Shape Key. This Operator is still useful for its ability to be run without
# swapping into Edit mode.
@set_operator_description_from_doc
class ApplyShapeKeyToReferenceKey(OperatorBase):
    """
    Applies the active shape key at its current strength (if non-zero) to the reference shape key (usually 'Basis'), and
    then inverts the active shape key so that it reverts its application.
    """
    # Blender 3.3 is the current oldest LTS and still has a small maximum length for Operator descriptions, so the
    # description can't be too long.
    bl_idname = "mysteryem.apply_shape_key_to_reference_key"
    # The label uses "Basis" because that is what most users are familiar with, though the reference Shape Key can have
    # any name.
    bl_label = "Apply Shape Key to Basis"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context: Context):
        obj = context.object

        if not obj:
            return cls.poll_fail("Requires an active Object")

        # Could be extended to other types that have Shape Keys, but only MESH supported for now
        if obj.type != 'MESH':
            return cls.poll_fail("The active Object must be a Mesh")

        if obj.mode == 'EDIT':
            return cls.poll_fail("The Mesh must not be in Edit mode")

        mesh = cast(Mesh, obj.data)
        shape_keys = mesh.shape_keys

        if not shape_keys:
            return cls.poll_fail("The Mesh must have Shape Keys")

        # The operator makes no sense when Shape Keys are in Absolute mode.
        if not shape_keys.use_relative:
            return cls.poll_fail("The Shape Keys must use the Relative mode")

        active_shape_key = obj.active_shape_key

        # Maybe possible if the active index is out of bounds?
        if not active_shape_key:
            return cls.poll_fail("An active Shape Key is required")

        reference_key = shape_keys.reference_key

        # If the active Shape Key is the Reference Key, then is does nothing because it is the base state of the Mesh.
        # Technically it can be relative to another Shape Key still, but this is not shown in the UI. For the purposes
        # of this operator, the Reference Key is treated as if it is always relative to itself.
        if active_shape_key == shape_keys.reference_key:
            return cls.poll_fail("The active Shape Key must be a different Shape Key to the reference Shape Key ('%s')"
                                 % reference_key.name)

        # If the active Shape Key is relative to itself, then it does nothing.
        if active_shape_key.relative_key == active_shape_key:
            return cls.poll_fail("The active Shape Key must not be relative to itself")

        return True

    def execute(self, context: Context):
        # If an object other than the active object is to be used, it can be specified using a context override.
        obj = context.object

        me = cast(Mesh, obj.data)

        # Get Shape Key which will be the new Reference Key. The `poll` function prevents this from being the current
        # Reference Key.
        new_reference_shape_key = obj.active_shape_key

        # Get sets of all Shape Keys which are recursively relative to the current Reference Key and the new reference
        # key.
        # The Shape Keys may be in their own sets if they are relative to themselves.
        relative_to_old_reference, relative_to_new_reference = get_recursive_relative_shape_keys(
            me.shape_keys, new_reference_shape_key)

        # Cancel execution if the new reference Shape Key is relative to itself (via a loop, since `poll` already
        # returns False for being immediately relative to itself since that will always do nothing).
        # If the Relative Keys loop back around, then if the key is turned into its reverse after applying, it would
        # affect all Shape Keys that it is relative to. Consider:
        # Key1 relative -> Key2
        # Key2 relative -> Key1
        # If Key1 is applied to the Reference Key, Key1 should be changed to a reversed Shape Key in order to undo the
        # application.
        # Since Key2 is relative to Key1, it has to be modified to account for the change in Key1 so that its relative
        # movement to Key1 stays the same.
        # Since Key1 is relative to Key2, it has to be modified to account for the change in Key2 so that its relative
        # movement to Key2 stays the same, but that creates an infinite loop.
        #
        # Another way of looking at it is if Key1 moves a vertex by +1, then Key2 MUST move that same vertex by -1 since
        # the two Shape Keys are relative to one other.
        # If Key1 is applied to the Reference Key, it should become a reversed Shape Key that moves a Vertex by -1
        # instead so that when it is re-applied, it undoes the initial application, but that would mean that Key2 would
        # have to become a Shape Key that moves a vertex by +1, but Key2 must keep its original relative movement of -1.
        if new_reference_shape_key in relative_to_new_reference:
            self.report({'ERROR_INVALID_INPUT'},
                        "Shape Key '%s' is recursively relative to itself, so cannot be applied"
                        % new_reference_shape_key.name)
            return {'CANCELLED'}

        apply_new_reference_key(obj=obj,
                                new_ref=new_reference_shape_key,
                                keys_relative_to_new_ref=relative_to_new_reference,
                                keys_relative_to_old_ref=relative_to_old_reference)

        # The active key is now the reverse of what it was before so rename it as such.
        reversed_string = " - Reversed"
        reversed_string_len = len(reversed_string)
        old_name = new_reference_shape_key.name

        if old_name[-reversed_string_len:] == reversed_string:
            # If the last letters of the name are `reversed_string`, remove them.
            new_reference_shape_key.name = old_name[:-reversed_string_len]
        else:
            # Add `reversed_string` to the end of the name, so it is clear that this Shape Key is now the reverse of what
            # it was before.
            new_reference_shape_key.name = old_name + reversed_string

        # Setting `.value` to `0.0` will make the Mesh appear unchanged in overall shape and help to show that the
        # Operator has worked correctly.
        if new_reference_shape_key.slider_min > 0.0:
            new_reference_shape_key.slider_min = 0.0
        new_reference_shape_key.value = 0.0
        new_reference_shape_key.slider_min = 0.0
        # Regardless of what the max was before, 1.0 will now fully undo the applied Shape Key.
        new_reference_shape_key.slider_max = 1.0

        self.report({'INFO'}, "Applied '%s' to the Reference Key" % old_name)
        return {'FINISHED'}


# Implementation


def get_recursive_relative_shape_keys(all_shape_keys: Key,
                                      new_ref_key: ShapeKey) -> tuple[set[ShapeKey], set[ShapeKey]]:
    """
    Get all Shape Keys which are recursively relative the current Reference Key and the specified new Reference Key.
    :param all_shape_keys: Shape Key data block.
    :param new_ref_key:
    :return:
    """
    assert new_ref_key.name in all_shape_keys.key_blocks

    # `shape_key.relative_key` gets the Shape Key that `shape_key` is directly relative to.
    # `reverse_relative_map` stores all the Shape Keys that are directly relative to each Shape Key.
    reverse_relative_map = {}

    current_ref_key = all_shape_keys.reference_key
    for shape_key in all_shape_keys.key_blocks:

        # Special handling for the reference Shape Key to treat it as if its always relative to itself.
        relative_key = current_ref_key if shape_key == current_ref_key else shape_key.relative_key
        keys_relative_to_relative_key = reverse_relative_map.get(relative_key)
        if keys_relative_to_relative_key is None:
            keys_relative_to_relative_key = {shape_key}
            reverse_relative_map[relative_key] = keys_relative_to_relative_key
        else:
            keys_relative_to_relative_key.add(shape_key)

    # Find all Shape Keys that are either directly or recursively relative to the Reference Key, and the same for
    # `new_ref_key`.

    # Pretty much a depth-first search, but with loop prevention.
    def inner_recursive_loop(key, shape_set, checked_set=None):
        if checked_set is None:
            checked_set = set()

        # Prevent infinite loops by maintaining a set of Shape Keys that have been checked.
        if key not in checked_set:
            # The current Shape Key must be added to the set before the recursive call.
            checked_set.add(key)
            keys_relative_to_shape_key_inner = reverse_relative_map.get(key)
            if keys_relative_to_shape_key_inner:
                for relative_to_inner in keys_relative_to_shape_key_inner:
                    shape_set.add(relative_to_inner)
                    inner_recursive_loop(relative_to_inner, shape_set, checked_set)

    recursively_relative_to_current_ref_key = set()
    # It should work to pick a different Shape Key to apply to, so long as that Shape Key is immediately relative to
    # itself (`key.relative_key == key`). For now, the reference Shape Key is always used.
    inner_recursive_loop(all_shape_keys.reference_key, recursively_relative_to_current_ref_key)

    recursively_relative_to_new_ref_key = set()
    inner_recursive_loop(new_ref_key, recursively_relative_to_new_ref_key)

    return recursively_relative_to_current_ref_key, recursively_relative_to_new_ref_key


def vertex_group_weight_generator(mesh: Mesh, vertex_group_index: int) -> Iterator[float]:
    """
    Generator for Vertex Group weights. Yields 0.0 when a Vertex does not belong to the Vertex Group.
    :param mesh:
    :param vertex_group_index:
    :return:
    """
    # Blender has no efficient way to get all the weights for a particular Vertex Group.
    # It's pretty much always a few times faster to create a new Shape Key from mix and get its "co" with
    # #foreach_get, but that requires temporarily muting all other Shape Keys and disabling `Object.show_only_shape_key`
    # to ensure the new shape is created with the correct mix.
    # For simplicity, and because Vertex Groups are rarely used on Shape Keys, the slower access of Vertex Group weights
    # is used here.
    # https://developer.blender.org/D6227 has the sort of function that is needed, which could make it into Blender one
    # day.
    # There has also been talk of Vertex Groups possibly becoming more like generic Mesh Attributes in the future, which
    # would hopefully be able to use some, or all, of the Attributes API.

    # This method for getting Vertex Group weights scales poorly with lots of vertex groups assigned to each vertex.
    #
    # An alternative method is to use `VertexGroup.weight(vertex_index)`, which does not scale poorly. However, it
    # raises an Error when the vertex index is not in the Vertex Group, and relying on catching the Error is really
    # slow.
    for v in mesh.vertices:
        for g in v.groups:
            if g.group == vertex_group_index:
                yield g.weight
                break
        else:
            # No matching vertex group index found.
            yield 0.0


# Figures out what needs to be added to each affected Shape key, then iterates through all the affected Shape keys,
# getting the current coordinates, adding the corresponding amount to it and then setting that as the new coordinates.
# Gets and sets Shape Key coordinates manually with #foreach_get and #foreach_set.
# The slowest part of this function when the number of Vertices increase are the #foreach_set and #foreach_get calls, so
# the number of calls of those should be minimised for performance.
def apply_new_reference_key(obj: Object,
                            new_ref: ShapeKey,
                            keys_relative_to_new_ref: set[ShapeKey],
                            keys_relative_to_old_ref: set[ShapeKey]):
    value = new_ref.value
    if value == 0.0:
        # 0.0 would have no effect, so set to 1.0.
        value = 1.0

    mesh = cast(Mesh, obj.data)
    num_verts = len(mesh.vertices)

    new_ref_key_vertex_group_name = new_ref.vertex_group
    if new_ref_key_vertex_group_name:
        new_ref_key_vertex_group = obj.vertex_groups.get(new_ref_key_vertex_group_name)
    else:
        new_ref_key_vertex_group = None

    new_ref_key_affected_by_own_application = new_ref in keys_relative_to_old_ref

    # Array of Vector type is flattened by #foreach_get into a sequence so the length needs to be multiplied by 3.
    flat_co_length = num_verts * 3

    # Store Shape Key coordinates for `new_ref`.
    # There is no need to initialise the elements to anything since they will all be overwritten.
    # The `ShapeKeyPoint` type's "co" property is a FloatProperty type, these are single precision floats.
    # It is extremely important for performance that the correct float type (np.single/np.float32) is used.
    # Using the wrong type could result in 3-5 times slower performance (depending on array length) due to Blender being
    # required to iterate through each element in the data first instead of immediately setting/getting all the data
    # directly.
    # See #foreach_getset in bpy_rna.cc of the Blender source for the implementation.
    new_ref_co_flat = np.empty(flat_co_length, dtype=np.single)
    new_ref_relative_key_co_flat = np.empty(flat_co_length, dtype=np.single)

    new_ref.data.foreach_get("co", new_ref_co_flat)
    new_ref.relative_key.data.foreach_get("co", new_ref_relative_key_co_flat)

    # This is movement of `new_ref` at a value of 1.0.
    difference_co_flat = np.subtract(new_ref_co_flat, new_ref_relative_key_co_flat)

    # Scale the difference based on the `.value` of `new_ref`.
    difference_co_flat_value_scaled = np.multiply(difference_co_flat, value)

    # These arrays can be reused over and over instead of allocating new arrays each time.
    temp_co_array = np.empty(flat_co_length, dtype=np.single)
    temp_co_array2 = np.empty(flat_co_length, dtype=np.single)

    if new_ref_key_vertex_group:
        # Scale the difference based on the Vertex Group.
        vertex_weight_generator = vertex_group_weight_generator(mesh, new_ref_key_vertex_group.index)
        vertex_group_weights = np.fromiter(vertex_weight_generator, difference_co_flat_value_scaled.dtype, num_verts)

        # Both arrays must be promoted to 2D views so that broadcasting can occur due to there being only a single
        # Vertex Group weight per vector.
        difference_co_flat_scaled = np.multiply(
            difference_co_flat_value_scaled.reshape(num_verts, 3),
            vertex_group_weights.reshape(num_verts, 1)
        ).ravel()
    else:
        difference_co_flat_scaled = difference_co_flat_value_scaled

    if new_ref_key_affected_by_own_application:
        # All Shape Keys in `keys_relative_to_new_ref` must also be in `keys_relative_to_old_ref`.
        # All the keys that will have only difference_co_flat_scaled added to them are those which are neither
        # `new_ref` nor relative to `new_ref`.
        keys_not_relative_to_new_ref_and_not_new_ref = (keys_relative_to_old_ref - keys_relative_to_new_ref) - {new_ref}

        # This for loop is where most of the execution will happen for typical setups of many Shape Keys directly
        # relative to the Reference Key.
        #
        # Add the difference between `new_ref` and `new_ref.relative_key` (scaled according to the `.value` and
        # `.vertex_group` of `new_ref`).
        # The coordinates for `new_ref.relative_key` has already been retrieved, so do it separately to save a
        # #foreach_get call.
        new_ref.relative_key.data.foreach_set(
            "co",
            np.add(new_ref_relative_key_co_flat, difference_co_flat_scaled, out=temp_co_array)
        )
        # And now the rest of the Shape Keys
        for key_block in keys_not_relative_to_new_ref_and_not_new_ref - {new_ref.relative_key}:
            key_block.data.foreach_get("co", temp_co_array)
            key_block.data.foreach_set("co", np.add(temp_co_array, difference_co_flat_scaled, out=temp_co_array))

        # Shorthand key:
        # NB = new_ref
        # NB.r = new_ref.relative_key
        # r(NB) = reversed(new_ref)
        # r(NB).r = reversed(new_ref).relative_key
        # NB.v = new_ref.value (`value`)
        # NB.vg = new_ref.vertex_group
        #
        # The difference between r(NB) and r(NB).r needs to be the negative of:
        #   (r(NB) - r(NB).r) * NB.vg = -((NB - NB.r) * NB.v * NB.vg)
        #                             = -(NB - NB.r) * NB.v * NB.vg
        # NB.vg cancels on both sides, leaving:
        #   r(NB) - r(NB).r = -(NB - NB.r) * NB.v
        # Rearranging for r(NB) gives:
        #   r(NB) = r(NB).r - (NB - NB.r) * NB.v
        # Note that (NB - NB.r) * NB.v = difference_co_flat_value_scaled so:
        #   r(NB) = r(NB).r - difference_co_flat_value_scaled
        # Note that r(NB).r = (NB.r + difference_co_flat_scaled) because that has been added to it.
        #   r(NB) = NB.r + difference_co_flat_scaled - difference_co_flat_value_scaled
        # Note that r(NB) = NB + X where X is what needs to be found to add to NB (and all Shape Keys relative to NB so
        # that their relative differences remain the same).
        #   NB + X = NB.r + difference_co_flat_scaled - difference_co_flat_value_scaled
        #   X = NB.r - NB + difference_co_flat_scaled - difference_co_flat_value_scaled
        #   X = -(NB - NB.r) + difference_co_flat_scaled - difference_co_flat_value_scaled
        # Fully expanding out would give:
        #   X = -(NB - NB.r) + (NB - NB.r) * NB.v * NB.vg - (NB - NB.r) * NB.v
        #
        # TODO: NB.vg is calculated now to simplify code, despite it being slower, so it may be possible to factorize
        #  better now. e.g.
        #   X = -(NB - NB.r) + (NB - NB.r) * NB.v * NB.vg - (NB - NB.r) * NB.v
        #   X = (NB - NB.r) * (-1 + 1 * NB.v * NB.vg - 1 * NB.v)
        #   X = (NB - NB.r) * (-1 + NB.v * NB.vg - NB.v)
        #   X = (NB - NB.r) * (-1 + NB.v(NB.vg - 1))
        #   X = difference_co_flat * (-1 + NB.v(NB.vg - 1))
        # OUTDATED:
        # In the case of there being a vertex group, it is too costly to calculate NB.vg on its own, so we will leave it
        # at:
        #   X = -(NB - NB.r) + difference_co_flat_scaled - (NB - NB.r) * NB.v
        #   Which we can either factor to
        #       X = (NB - NB.r)(-1 - NB.v) + difference_co_flat_scaled
        #       X = difference_co_flat * (-1 - NB.v) + difference_co_flat_scaled
        #   Or, as NB - NB.r = difference_co_flat, calculate as
        #       X = -difference_co_flat + difference_co_flat_scaled - difference_co_flat_value_scaled
        #
        # The numpy functions take close to a negligible amount of the total function time, so the choice is not very
        # important, however, from my own benchmarks, `np.multiply(array1, scalar, out=output_array)` starts to scale
        # slightly better than `np.add(array1, array2, out=output_array)` once array1 gets to around 9000 elements or
        # more.
        # I guess this is due to the fact that the add operation needs to do 1 extra array access per element, and that
        # eventually surpasses the effect of the multiply operation being more expensive than the add operation.
        # In this case, the array length is 3*num_verts, meaning the multiplication option gets better at around 3000
        # Vertices, so the multiplication option is what is used here.
        if new_ref_key_vertex_group:
            np.multiply(difference_co_flat, -1 - new_ref.value, out=temp_co_array2)
            np.add(temp_co_array2, difference_co_flat_scaled, out=temp_co_array2)

            # The coordinates for `new_ref` have already been acquired, so `new_ref` can be done separately from the
            # others to save a #foreach_get call.
            new_ref.data.foreach_set("co", np.add(new_ref_co_flat, temp_co_array2, out=temp_co_array))

            # Now add to the rest of the keys
            for key_block in keys_relative_to_new_ref:
                key_block.data.foreach_get("co", temp_co_array)
                key_block.data.foreach_set("co", np.add(temp_co_array, temp_co_array2, out=temp_co_array))
        # But for there not being a vertex group, the NB.vg term can be eliminated as it becomes effectively 1.0:
        #   X = -(NB - NB.r) + (NB - NB.r) * NB.v - (NB - NB.r) * NB.v
        # Then the last part cancels out:
        #   X = -(NB - NB.r)
        # Giving X = -difference_co_flat.
        else:
            # Instead of adding the difference_co_flat_scaled to each Shape Key it will be subtracted from each Shape
            # Key instead.
            # The coordinates for `new_ref` have already been acquired, so it can be done separately to avoid a
            # #foreach_get call.
            # Note that:
            #   difference_co_flat = NB - NB.r
            # Rearrange for NB.r:
            #   NB.r = NB - difference_co_flat
            # Instead of doing `np.subtract(new_ref_co_flat, difference_co_flat)`, NB can simply be set to NB.r.
            new_ref.data.foreach_set("co", new_ref_relative_key_co_flat)
            # And the rest of the Shape Keys
            for key_block in keys_relative_to_new_ref:
                key_block.data.foreach_get("co", temp_co_array)
                key_block.data.foreach_set("co", np.subtract(temp_co_array, difference_co_flat, out=temp_co_array))
    else:
        # `new_ref` is not relative to the Reference Key so the Shape Keys `new_ref` is relative to will remain
        # unchanged.
        # Shape Keys relative to the Reference Key and Shape Keys recursively relative to `new_ref` will be mutually
        # exclusive.
        # Typical user setups have all the Shape Keys immediately relative to the Reference Key, so this will not be
        # used much.

        # Add the difference between `new_ref` and `new_ref.relative_key` (scaled according to the `.value` and
        # `.vertex_group` of `new_ref`).
        for key_block in keys_relative_to_old_ref:
            key_block.data.foreach_get("co", temp_co_array)
            key_block.data.foreach_set("co", np.add(temp_co_array, difference_co_flat_scaled, out=temp_co_array))

        # The difference between the reversed Shape Key and its Relative Key needs to equal the negative of the
        # difference between `new_ref` and `new_ref`.relative_key multiplied.
        # `new_ref.vertex_group` should be present on both.
        #   (r(NB) - r(NB).r) * NB.vg = -((NB - NB.r) * NB.v * NB.vg)
        #                             = -(NB - NB.r) * NB.v * NB.vg
        # NB.vg cancels on both sides, leaving:
        #   r(NB) - r(NB).r = -(NB - NB.r) * NB.v
        # r(NB).r is unchanged, meaning r(NB).r = NB.r
        #   r(NB) - NB.r = -(NB - NB.r) * NB.v
        # r(NB) = X + NB where X is what needs to be found to add
        #   X + NB - NB.r = -(NB - NB.r) * NB.v
        # Rearrange for X:
        #   X = -(NB - NB.r) - (NB - NB.r) * NB.v
        #
        # (NB - NB.r) can be factorised:
        #   X = (NB - NB.r)(-1 - NB.v)
        # Note that (NB - NB.r) is difference_co_flat, giving:
        #   X = difference_co_flat * (-1 - NB.v)
        #
        # Alternatively, instead of factorising, note that (NB - NB.r) * NB.v is difference_co_flat_value_scaled:
        #   X = -(NB - NB.r) - difference_co_flat_value_scaled
        # Note that (NB - NB.r) is difference_co_flat, giving
        #   X = -difference_co_flat - difference_co_flat_value_scaled
        # Or
        #   X = -(difference_co_flat + difference_co_flat_value_scaled)
        #
        # Since NB.vg is not present, it does not matter whether `new_ref` has a Vertex Group or not.
        #
        # As with before, the multiplication option is used due to it scaling slightly better with a larger number of
        # Vertices:
        # X = difference_co_flat * (-1 - NB.v)
        np.multiply(difference_co_flat, -1 - new_ref.value, out=temp_co_array2)

        # The coordinates for `new_ref` have already been acquired, so it can be done separately from the others to save
        # a #foreach_get call.
        new_ref.data.foreach_set("co", np.add(new_ref_co_flat, temp_co_array2, out=temp_co_array))
        # And now the rest of the Shape Keys.
        for key_block in keys_relative_to_new_ref:
            key_block.data.foreach_get("co", temp_co_array)
            key_block.data.foreach_set("co", np.add(temp_co_array, temp_co_array2, out=temp_co_array))

    # Update Mesh Vertices to avoid reference Shape Key and Mesh Vertices being desynced until Edit mode has been
    # entered and exited, which can cause odd behaviour when creating Shape Keys with `from_mix=False`, when removing
    # all Shape Keys or exporting as a format that supports exporting Shape Keys.
    mesh.shape_keys.reference_key.data.foreach_get("co", temp_co_array)
    mesh.vertices.foreach_set("co", temp_co_array)


# Registration

def draw_in_menu(self, context: Context):
    self.layout.separator()
    # CATS used the KEY_HLT icon for years, so it may be best to leave the icon as is because people are used to it.
    self.layout.operator(ApplyShapeKeyToReferenceKey.bl_idname, icon='KEY_HLT')


# Links for Right Click -> Online Manual.
def add_manual_map():
    url_manual_prefix = "https://github.com/Mysteryem/blender-apply-shape-key-to-basis"
    url_manual_mapping = (
        ("bpy.ops." + ApplyShapeKeyToReferenceKey.bl_idname, ""),
    )
    return url_manual_prefix, url_manual_mapping


def register():
    bpy.utils.register_class(ApplyShapeKeyToReferenceKey)
    bpy.utils.register_manual_map(add_manual_map)
    bpy.types.MESH_MT_shape_key_context_menu.append(draw_in_menu)


def unregister():
    bpy.types.MESH_MT_shape_key_context_menu.remove(draw_in_menu)
    bpy.utils.unregister_manual_map(add_manual_map)
    bpy.utils.unregister_class(ApplyShapeKeyToReferenceKey)


# For testing in Blender Text Editor.
if __name__ == "__main__":
    # Try and unregister the previously registered version.
    unregister_attribute = "shape_key_to_basis_unregister_old"
    temp_storage = bpy.types.WindowManager
    if old_unregister := getattr(temp_storage, unregister_attribute, None):
        try:
            old_unregister()
        except Exception as e:
            print(e)
    register()
    setattr(temp_storage, unregister_attribute, unregister)
