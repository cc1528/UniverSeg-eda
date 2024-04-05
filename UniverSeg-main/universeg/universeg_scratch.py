from torch.utils.data import DataLoader

# Assuming you have your dataset and loaders defined

# Define your dataset and loaders
# train_dataset = YourDataset(...)
# val_dataset = YourDataset(...)
# test_dataset = YourDataset(...)

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size)
# test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Initialize model without pretrained weights
model = universeg(pretrained=False)

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        inputs, targets = batch['input'], batch['target']
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch['input'], batch['target']
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss /= len(val_loader.dataset)
    accuracy = 100.0 * correct / total

    print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {accuracy:.2f}%")

# Test
model.eval()
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        inputs, targets = batch['input'], batch['target']
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

test_loss /= len(test_loader.dataset)
accuracy = 100.0 * correct / total

print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
