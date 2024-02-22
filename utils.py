
import torch
from torchvision import transforms

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def transform(examples):
    examples['image'] = [preprocess(image.convert("RGB")) for image in examples['image']]
    return examples

def collate_fn(examples):
    images = []
    labels = []
    for example in examples:
        images.append((example['image']))
        labels.append(example['label'])

    images = torch.stack(images)
    labels = torch.tensor(labels)
    return {'image': images, 'label': labels}
