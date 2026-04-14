# Restraint Force on Resume

ALWAYS add restraint force (k=0) to the OpenMM system before calling
`loadState()`, even on resume when no restraints are active.

Without it, `loadState()` fails because the serialized state references
a force that doesn't exist in the system.
