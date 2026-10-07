import torch

from src.v4_model import ResNet50CustomV4


NUM_CLASSES = 24
IMAGE_SIZE = 224


print("=" * 60)
print("V4 MODEL ARCHITECTURE TEST")
print("=" * 60)

# ------------------------------------------------------------
# 1. Create model
# ------------------------------------------------------------

model = ResNet50CustomV4(
    num_classes=NUM_CLASSES,
    pretrained=True,
)

print("\nModel created successfully.")


# ------------------------------------------------------------
# 2. Test forward pass
# ------------------------------------------------------------

x = torch.randn(
    1,
    3,
    IMAGE_SIZE,
    IMAGE_SIZE,
)

with torch.no_grad():
    output = model(x)

print(f"\nInput shape:  {tuple(x.shape)}")
print(f"Output shape: {tuple(output.shape)}")


# ------------------------------------------------------------
# 3. Verify output
# ------------------------------------------------------------

assert output.shape == (1, NUM_CLASSES), (
    f"Expected output shape (1, {NUM_CLASSES}), "
    f"got {tuple(output.shape)}"
)

print("\n✓ Output shape is correct.")


# ------------------------------------------------------------
# 4. Phase 1 test
# ------------------------------------------------------------

model.freeze_backbone()

phase1_trainable = [
    name
    for name, parameter in model.named_parameters()
    if parameter.requires_grad
]

phase1_frozen = [
    name
    for name, parameter in model.named_parameters()
    if not parameter.requires_grad
]

print("\n" + "-" * 60)
print("PHASE 1")
print("-" * 60)

print(f"Trainable parameters: {len(phase1_trainable)}")
print(f"Frozen parameters:    {len(phase1_frozen)}")

print("\nTrainable:")
for name in phase1_trainable:
    print(f"  ✓ {name}")

# Verify ResNet backbone is frozen.
assert all(
    not parameter.requires_grad
    for parameter in model.backbone.parameters()
)

# Verify custom layers are trainable.
for module in [
    model.conv1,
    model.conv2,
    model.linear1,
    model.linear2,
]:
    assert all(
        parameter.requires_grad
        for parameter in module.parameters()
    )

print("\n✓ Phase 1 freezing is correct.")


# ------------------------------------------------------------
# 5. Phase 2 test
# ------------------------------------------------------------

model.unfreeze_layer4()

print("\n" + "-" * 60)
print("PHASE 2")
print("-" * 60)

# ResNet50 layer4 should be trainable.
assert all(
    parameter.requires_grad
    for parameter in model.backbone[7].parameters()
)

# ResNet50 layers 1-3 should remain frozen.
for layer_index in [0, 1, 2, 3, 4, 5, 6]:
    assert all(
        not parameter.requires_grad
        for parameter in model.backbone[layer_index].parameters()
    )

# Custom layers should remain trainable.
for module in [
    model.conv1,
    model.conv2,
    model.linear1,
    model.linear2,
]:
    assert all(
        parameter.requires_grad
        for parameter in module.parameters()
    )

phase2_trainable = [
    name
    for name, parameter in model.named_parameters()
    if parameter.requires_grad
]

print(f"Trainable parameters: {len(phase2_trainable)}")

print("\nTrainable:")
for name in phase2_trainable:
    print(f"  ✓ {name}")

print("\n✓ Phase 2 freezing is correct.")


# ------------------------------------------------------------
# 6. Final test
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("ALL V4 MODEL TESTS PASSED")
print("=" * 60)