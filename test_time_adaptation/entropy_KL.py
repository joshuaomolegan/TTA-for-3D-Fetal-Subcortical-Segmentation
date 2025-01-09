import torch
import torch.jit
import SimpleITK as sitk
import adaptation_base

# Note this class uses the functions in adaptation_base.py, other than softmax_entropy
class EntropyKL(adaptation_base.BaseAdaptation):
    def __init__(self, model, optimizer, atlas_labels_path, steps=1, lambd=1.0, episodic=False):
        super().__init__(model=model, optimizer=optimizer, loss=lambda x: entropy_KL_loss(x, atlas_labels_path, lambd=lambd), 
                         steps=steps, episodic=episodic)


# Load Atlas and calculate class ratios
def load_atlas_labels(atlas_labels_path):
    atlas_labels = sitk.GetArrayFromImage(sitk.ReadImage(atlas_labels_path))
    atlas_labels = torch.from_numpy(atlas_labels).long()
    return atlas_labels

'''
Calculate class ratios
Note the atlas has additional classes (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
The model predicts classes 0 to 4 in the order background, choroidPlexus, ventricle, cavum, cerebellum
In the atlas, these correspond to classes 4, 5, 2, 3 respectively. See .../struc_index.txt for more info
Note in both the Atlas and the model, class 0 is the background
'''
def calculate_atlas_class_ratios(label_vol, ordered_class_indices = [4, 5, 2, 3]):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # Create a tensor to store the class counts in shape (batch, num_classes)
    class_counts = torch.zeros((1, len(ordered_class_indices)), device=device)
    
    for i in range(len(ordered_class_indices)):
        # Sum over all except the batch dimension
        mask = (label_vol == ordered_class_indices[i])

        class_counts[:, i] = torch.sum(mask)
    return class_counts / (torch.sum(class_counts, dim=1, keepdim=True)) # Divide each class sum by the sum of all class sums

# Note in our use case (5 classes, 160 x 160 x 160 volumes), the model outputs are size [batch, 5, 160, 160, 160]
def calculate_model_class_ratios(preds_vol, num_classes = 5, omit_background=False):
    if omit_background:
        num_classes -= 1

    start_index = 1 if omit_background else 0

    class_sum = torch.sum(preds_vol[:, start_index:], dim=(2, 3, 4)) # Sum of all probabilities for each class i.e. sum over all voxels
    class_sum = class_sum.unsqueeze(0) # Add a dimension to match the shape of the atlas class ratios
    
    return class_sum / (torch.sum(class_sum, dim=2, keepdim=True)) # Divide each class sum by the sum of all class sums

# Calculate KL divergence between atlas and model prediction labels
def calculate_KL_divergence(model_output, atlas_class_ratios, num_classes=5):
    model_class_ratios = calculate_model_class_ratios(model_output, num_classes, omit_background=True)
    eps = 1e-10

    atlas_class_ratios = atlas_class_ratios.unsqueeze(1) # Add a dimension to match the shape of the model class ratios
    return torch.sum(atlas_class_ratios * torch.log((atlas_class_ratios / (model_class_ratios + eps)) + eps), dim=2) # Sum over all

# In the paper, lambda is set to 100
def entropy_KL_loss(x: torch.Tensor, atlas_labels_path: str, lambd: float = 1.0) -> torch.Tensor:
    """Entropy of softmax distribution from logits + KL divergence of model and atlas class ratios"""

    # v_k as defined in the paper
    model_class_ratios = calculate_model_class_ratios(x.softmax(1))
    v_k = torch.pow(model_class_ratios, -1)
    v_k = v_k / torch.sum(v_k, dim=1) # Divide each class sum by the sum of all class sums
    v_k = v_k[0]
    assert v_k.shape[1] == x.shape[1] # Check that the number of classes in v_k is equal to the number of classes in x (excluding the background class)
    v_k = v_k.unsqueeze(2).unsqueeze(3).unsqueeze(4) # Add dimensions to match x.shape
    
    mean_softmax_entropy = torch.mean(-(v_k * x.softmax(1) * x.log_softmax(1)).sum(1), dim=(1, 2, 3))

    atlas_labels = load_atlas_labels(atlas_labels_path)

    atlas_class_ratios = calculate_atlas_class_ratios(atlas_labels)
    
    kl = calculate_KL_divergence(x.softmax(1), atlas_class_ratios)
    
    return mean_softmax_entropy + lambd * torch.sum(kl, dim=0)