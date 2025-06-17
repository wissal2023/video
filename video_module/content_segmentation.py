# content_segmentation.py
import re
import torch
from pathlib import Path
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer
import torch.nn as nn

class ContentSegmenter:
    """
    Segmente le contenu en sections optimales pour les diapositives
    """
    
    def __init__(self):
        self.model_path = Path("models/content_segmenter.joblib")
        self.embedding_model = None
        self.kmeans_model = None
        try:
            self.embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        except Exception as e:
            print(f"?? Erreur chargement modèle d'embeddings: {e}")
            self.embedding_model = None
    
    def segment_content(self, content, max_slides=20, min_content_per_slide=100):
        if not self.embedding_model:
            return self.segment_by_headings(content)
        segments = self.segment_by_semantic_similarity(content, max_slides, min_content_per_slide)
        if not segments:
            return self.segment_by_headings(content)
        return segments
    
    def segment_by_headings(self, content):
        segments = []
        heading_pattern = re.compile(r'^(#+)\s+(.*?)$', re.MULTILINE)
        headings = [(match.group(1), match.group(2), match.start()) for match in heading_pattern.finditer(content)]
        image_pattern = re.compile(r'!\[(.*?)\]\((.*?)\)')
        images = [(match.start(), match.group(0)) for match in image_pattern.finditer(content)]
        if not headings:
            has_image = bool(images)
            segments.append({
                'title': "Document",
                'content': content,
                'has_image': has_image
            })
        else:
            for i, (level, title, start) in enumerate(headings):
                if i < len(headings) - 1:
                    end = headings[i+1][2]
                else:
                    end = len(content)
                section_content = content[start:end]
                has_image = any(start <= img_pos < end for img_pos, _ in images)
                segments.append({
                    'title': title,
                    'content': section_content,
                    'has_image': has_image
                })
        return segments
    
    def segment_by_semantic_similarity(self, content, max_slides, min_content_per_slide):
        try:
            paragraphs = re.split(r'\n\n+', content)
            paragraphs = [p.strip() for p in paragraphs if p.strip()]
            if not paragraphs:
                return self.segment_by_headings(content)
            titles = []
            title_indices = []
            for i, p in enumerate(paragraphs):
                if re.match(r'^#+\s+', p):
                    titles.append(p)
                    title_indices.append(i)
            embeddings = self.embedding_model.encode(paragraphs)
            if len(paragraphs) <= max_slides or len(paragraphs) <= len(titles) * 2:
                return self.segment_by_headings(content)
            n_clusters = min(max_slides, max(2, len(titles)))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(embeddings)
            cluster_paragraphs = {}
            for i, cluster_id in enumerate(clusters):
                if cluster_id not in cluster_paragraphs:
                    cluster_paragraphs[cluster_id] = []
                cluster_paragraphs[cluster_id].append(i)
            ordered_clusters = []
            for cluster_id, paragraph_indices in cluster_paragraphs.items():
                min_idx = min(paragraph_indices)
                ordered_clusters.append((min_idx, cluster_id, paragraph_indices))
            ordered_clusters.sort()
            segments = []
            for _, cluster_id, paragraph_indices in ordered_clusters:
                paragraph_indices.sort()
                segment_title = "Section"
                for idx in paragraph_indices:
                    if idx in title_indices:
                        title_text = re.sub(r'^#+\s+', '', paragraphs[idx])
                        segment_title = title_text
                        break
                segment_content = "\n\n".join([paragraphs[i] for i in paragraph_indices])
                has_image = bool(re.search(r'!\[.*?\]\(.*?\)', segment_content))
                if len(segment_content) < min_content_per_slide and segments:
                    segments[-1]['content'] += "\n\n" + segment_content
                    continue
                segments.append({
                    'title': segment_title,
                    'content': segment_content,
                    'has_image': has_image
                })
            return segments
        except Exception as e:
            print(f"?? Erreur segmentation sémantique: {e}")
            return self.segment_by_headings(content)

class DeepContentSegmenter(ContentSegmenter):
    
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.embedding_model:
            self.embedding_model = self.embedding_model.to(self.device)
        self.autoencoder = None
       
        try:            
            class SegmentationAutoencoder(nn.Module):
                def __init__(self, input_size, hidden_size, num_clusters):
                    super(SegmentationAutoencoder, self).__init__()
                    self.encoder = nn.Sequential(
                        nn.Linear(input_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, num_clusters)
                    )
                    self.decoder = nn.Sequential(
                        nn.Linear(num_clusters, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, input_size)
                    )
                
                def forward(self, x):
                    embeddings = self.encoder(x)
                    reconstructed = self.decoder(embeddings)
                    return embeddings, reconstructed
            
            # Initialiser le modèle avec des dimensions adaptées
            self.autoencoder = SegmentationAutoencoder(384, 128, 20).to(self.device)
            print("? Modèle de segmentation deep learning initialisé")
            
        except Exception as e:
            print(f"?? Erreur initialisation autoencoder: {e}")
            self.autoencoder = None
            
