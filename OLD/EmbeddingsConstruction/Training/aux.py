import torch
import torch.nn.functional as F

# --------- Contrastive Loss ---------
def nt_xent_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    batch_size = z1.size(0)
    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z1.device)
    similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)

    positives = torch.sum(z1 * z2, dim=-1)
    positives = torch.cat([positives, positives], dim=0)
    logits = torch.cat([positives.unsqueeze(1), similarity_matrix], dim=1)
    labels = torch.zeros(2 * batch_size, dtype=torch.long).to(z1.device)

    return F.cross_entropy(logits / temperature, labels)