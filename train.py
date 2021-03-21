import numpy as np
import torch
import librosa
from src.dataset import EmodbDataset
from src.arch import NaiveCNN
from tqdm import tqdm

N_FFT = 2048
HOP_LENGTH = 512
N_MFCC = 13
NUM_CLASSES = 4

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-5

TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

PRECISION = 32
DEVICE = 'cuda'

VERBOSE = 2
EPOCHS = 10

PRECISION = torch.float16 if PRECISION == 16 else torch.float32

def collate_fn(batch):
    features, targets = zip(*batch)

    targets = torch.tensor(targets, dtype=torch.int64) # long tensor

    features = np.stack(features, axis=0)
    features = torch.from_numpy(features).float().unsqueeze(1)

    return features, targets

class Transform():

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        return librosa.feature.mfcc(audio, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mfcc=N_MFCC)

dataset_path = "./data/emodb"
ds = EmodbDataset(dataset_path, transform=Transform())

total_length = len(ds)

train_length = int(total_length * TRAIN_RATIO)
val_length = int(total_length * VAL_RATIO)
test_length = int(total_length * TEST_RATIO)

es_total_length = train_length + val_length + test_length

if es_total_length != total_length:
    train_length += (total_length-es_total_length)

train_ds, val_ds, test_ds = torch.utils.data.random_split(ds, [
    train_length, val_length, test_length])

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=4, collate_fn=collate_fn)

val_dl = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=2, collate_fn=collate_fn)

test_dl = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=2, collate_fn=collate_fn)

model = NaiveCNN(num_classes=NUM_CLASSES)
model.to(DEVICE, PRECISION)

loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY)

loss_bucket = []

for i in range(EPOCHS):
    # train loop

    model.train()
    for batch, targets in tqdm(train_dl, total=train_length//BATCH_SIZE, desc="Training Step"):
        # set gradients to zero
        optimizer.zero_grad()

        # model inference
        logits = model(batch.to(DEVICE, PRECISION))

        # calculate loss
        loss = loss_fn(logits, targets.to(DEVICE))

        # calculate gradients
        loss.backward()

        # update model parameters
        optimizer.step()

        loss_bucket.append(loss.detach().cpu())

        if len(loss_bucket) == VERBOSE:
            mean_loss = sum(loss_bucket) / len(loss_bucket)
            print("[{}/{}] Training Loss: {:.03f} ".format(i+1, EPOCHS, mean_loss))
            # reset the bucket
            loss_bucket = []

    # validation step
    model.eval()

    val_loss_bucket = []
    val_scores = []
    val_gts = []

    for batch, targets in tqdm(val_dl, total=val_length//BATCH_SIZE, desc="Validation Step"):
        with torch.no_grad():
            # model inference
            logits = model(batch.to(DEVICE, PRECISION))

            # calculate scores
            scores = torch.nn.functional.softmax(logits, dim=1)

            # calculate loss
            loss = loss_fn(logits, targets.to(DEVICE))

            val_loss_bucket.append(loss.item())
            val_scores.append(scores.cpu())
            val_gts.append(targets.cpu())
    val_scores = torch.cat(val_scores, dim=0)
    val_gts = torch.cat(val_gts, dim=0)

    val_preds = val_scores.argmax(dim=1)
    val_acc = (val_preds == val_gts).sum() / val_preds.shape[0]

    print("[{}/{}] Validation Loss : {:.03f} Accuracy: {:.03f}".format(
        i+1, EPOCHS, sum(val_loss_bucket) / len(val_loss_bucket), val_acc))

# test the final model
model.eval()

test_loss_bucket = []
test_scores = []
test_gts = []

for batch, targets in tqdm(test_dl, total=test_length//BATCH_SIZE, desc="Test Step"):
    with torch.no_grad():
        # model inference
        logits = model(batch.to(DEVICE, PRECISION))

        # calculate scores
        scores = torch.nn.functional.softmax(logits, dim=1)

        # calculate loss
        loss = loss_fn(logits, targets.to(DEVICE))

        test_loss_bucket.append(loss.item())
        test_scores.append(scores.cpu())
        test_gts.append(targets.cpu())

test_scores = torch.cat(test_scores, dim=0)
test_gts = torch.cat(test_gts, dim=0)

test_preds = test_scores.argmax(dim=1)
test_acc = (test_preds == test_gts).sum() / test_preds.shape[0]

print("[{}/{}] Test Loss: {:.03f} Accuracy: {:.03f}".format(
    i+1, EPOCHS, sum(test_loss_bucket) / len(test_loss_bucket), test_acc))