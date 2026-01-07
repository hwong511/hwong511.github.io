---
layout: post
title: Car Insurance Fraud Detector
description: predicting fraud from damage images using CNN and ViT
---

The Goal
============

Insurance fraud costs billions annually, with image evidence at the heart of most investigations. My team and I built ML models to detect potential fraud by analyzing car damage photographs. This matters because automating initial screening may help adjusters process claims faster while flagging suspicious cases for deeper review.

The Data
============

We used a dataset of car damage images from insurance claims, labeled as fraudulent or legitimate. The dataset included various angles and lighting conditions to simulate real-world scenarios.

We immediately noticed that in our dataset of 5,200 training images, only 200 were fraud. This extreme imbalance mirrors reality but creates a modeling challenge. We took 2 approaches to address this:

Approach 1: Vision Transformer with Class Weights
------------

We fine-tuned Google's ViT-Base-Patch16-224, leveraging its pre-trained ImageNet knowledge. To handle imbalance, we implemented weighted cross-entropy loss, penalizing the model 25x more for missing fraud cases. Training used AdamW optimizer with learning rate warmup and gradient clipping to handle the instability from extreme class weights.

~~~
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=len(label_map),
    id2label={v: k for k, v in label_map.items()},
    label2id=label_map,
    ignore_mismatched_sizes=True
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01, eps=1e-8)
criterion = nn.CrossEntropyLoss(weight=weights)
~~~

Approach 2: Custom CNN with Augmentation + Hyperparameter Tuning
------------

Our second approach built a CNN from scratch, attacking imbalance from multiple angles:

  * Selective augmentation: We applied random crops, flips, rotations, and color jittering only to fraud images, giving the model more diverse examples without collecting new data.

~~~
# Data augmentation transforms for training
train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1,0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Weighted sampling to handle class imbalance
counts = np.bincount([y for _,y in train_ds.samples])
class_weights = 1. / counts
sample_weights = [class_weights[y] for _,y in train_ds.samples]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
~~~

  * Systematic optimization: Using Optuna across 50 trials, we tuned learning rate, weight decay, dropout, and batch size to maximize F1-score. Best config: lr=0.00025, weight_decay=7.6e-6, dropout=0.615, batch_size=32.

~~~
def objective(trial):
    # Hyperparameter search space
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    dropout = trial.suggest_float("dropout_rate", 0.1, 0.7)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    
    # Build EfficientNet model with tunable dropout
    model = TunableNet(dropout).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).float().to(device))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Train for 3 epochs and evaluate
    for epoch in range(3):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
    
    # Evaluate on validation set
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X, y in val_loader:
            logits = model(X.to(device))
            preds.extend(logits.argmax(1).cpu().numpy())
            trues.extend(y.cpu().numpy())
    
    return f1_score(trues, preds, pos_label=1)

# Run optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
# Best: lr=0.00025, weight_decay=7.6e-6, dropout=0.615, batch_size=32
~~~

Results
============

![Model Performance Comparison]({{site.baseurl}}/assets/images/Vit_vs_CNN.png)

The ViT achieved higher accuracy, but the custom model won where it matters: catching fraud (96% recall vs. 75%) while maintaining near-perfect precision (99.5%). When the custom model predicted fraud, it was almost always right. The F1-score difference—0.9773 vs. 0.8434—quantified this superior balance.

We beleive this is because the targeted augmentation + careful hyperparameter tuning helped the custom model learn fraud patterns more robustly than the ViT's class weights alone.

Takeaways
============

  * Selective augmentation matters: Initially we augmented all images, but performance improved when we only transformed the minority class. This preserved the natural distribution of legitimate claims while enriching fraud examples.
  * Training stability: With 25x class weight differences, loss values swung wildly. Gradient clipping and warmup schedules were essential for stable training.
  * The right metrics: In fraud detection, recall and precision matter far more than accuracy. Understanding the business problem shapes architecture choices.
  * Computational reality: Training on CPU took 10-17 minutes per epoch. We optimized by disabling multiprocessing and managing memory carefully.
