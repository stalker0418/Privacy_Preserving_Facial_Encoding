from torchvision import transforms


class ImageTransformations:
    def __init__(self):
        pass

    def get_transform(self):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        return transform
    
    def get_transform_90_degrees(self):
        aug_transform_90 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(degrees=(90,90)),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        return aug_transform_90
    
    def get_transform_180_degrees(self):
        aug_transform_180 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(degrees=(180,180)),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        return aug_transform_180
    
    def get_transform_270_degrees(self):
        aug_transform_270 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(degrees=(270,270)),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        return aug_transform_270
    
    def get_transform_flip(self):
        aug_transform_flip = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        return aug_transform_flip
    
    def get_transform_random_flip(self):
        aug_transform_flip_random = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomRotation(degrees=(-20,20)),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        return aug_transform_flip_random
    
    def get_transform_random(self):
        aug_transform_random = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(degrees=(-20,20)),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        return aug_transform_random
    
    
