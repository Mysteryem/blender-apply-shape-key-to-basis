# Apply Shape Key to Basis Blender Add-on
Blender addon for applying the active shape key to the Basis (first) shape key and propagating the changes to the shape
keys that are relative to the Basis or recursively relative to the Basis.

The current Value (if non-zero) and Vertex Group (if set) of the Shape Key are taken into consideration.

The shape key that was applied is reversed such that applying it again will undo the original application.

Supports Blender 3.3 and newer. The add-on may work with older versions of Blender, but no support will be provided for
those versions.

![image](https://github.com/Mysteryem/blender-apply-shape-key-to-basis/assets/495015/399dfa7b-c98b-4937-81ca-fcf915c18945)

Editing the Basis shape key in Edit mode only propagates changes to the immediately relative shape keys in Blender 3.6
and older. This addon can be run from Object mode and can be used to correctly propagate changes to all shape keys
relative to the Basis.

For example:<br>
If `Key2` is relative to `Key1`<br>
and `Key1` is relative to `Basis`<br>
applying the active shape key to `Basis` propagates the change to both `Key1`  and `Key2`. 


Originally made for the Cats Blender Plugin to replace its slower implementation at the time. Now separated into its own
Add-on and made even faster (~40 times faster for meshes with 90,000 vertices).
