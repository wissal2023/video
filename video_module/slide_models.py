# slide_models.py
import joblib
import numpy as np
import re
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from PIL import Image, ImageStat

class SlideQualityModel:
    """
    Évalue la qualité des diapositives en utilisant l'apprentissage automatique traditionnel
    """
    
    def __init__(self):
        self.model_path = Path("models/slide_quality_model.joblib")
        self.model = None
        self.feature_names = None
        
        # Charger ou initialiser le modèle
        if self.model_path.exists():
            try:
                self.model = joblib.load(self.model_path)
                if hasattr(self.model, 'feature_names_in_'):
                    self.feature_names = self.model.feature_names_in_
                else:
                    sample_features = self.extract_slide_features("Sample content", "Sample title", True)
                    self.feature_names = list(sample_features.keys())
                print(f"? Modèle de qualité de diapositive chargé depuis {self.model_path}")
            except Exception as e:
                print(f"?? Erreur chargement modèle: {e}")
                self.initialize_model()
        else:
            self.initialize_model()
    
    def initialize_model(self):
        sample_features = self.extract_slide_features("Sample content", "Sample title", True)
        self.feature_names = list(sample_features.keys())
        self.model = RandomForestRegressor(n_estimators=50, random_state=42)
        print("? Nouveau modèle de qualité initialisé (non entraîné)")
    
    def extract_slide_features(self, slide_content, slide_title=None, has_image=False):
        features = {}
        content = re.sub(r'!\[.*?\]\(.*?\)', '', slide_content) if slide_content else ""
        features['text_length'] = len(content) if content else 0
        features['word_count'] = len(content.split()) if content else 0
        features['sentence_count'] = len(re.split(r'[.!?]+', content)) if content else 0
        features['avg_word_length'] = np.mean([len(w) for w in content.split()]) if content and content.split() else 0
        features['bullet_ratio'] = len(re.findall(r'[-•*]', content)) / max(1, len(content)) if content else 0
        features['paragraph_count'] = content.count('\n\n') + 1 if content else 0
        features['has_title'] = 1 if slide_title and len(slide_title) > 0 else 0
        features['has_image'] = 1 if has_image else 0
        return features
    
    def train(self, slide_data, quality_scores):
        X_features = []
        for slide in slide_data:
            features = self.extract_slide_features(
                slide.get('content', ''),
                slide.get('title', ''),
                slide.get('has_image', False)
            )
            X_features.append(list(features.values()))
        X = np.array(X_features)
        y = np.array(quality_scores)
        self.model.fit(X, y)
        y_pred = self.model.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        print(f"?? Modèle entraîné: MSE = {mse:.4f}, R² = {r2:.4f}")
        joblib.dump(self.model, self.model_path)
        print(f"?? Modèle sauvegardé dans {self.model_path}")
        return mse, r2
    
    def predict_quality(self, slide_content, slide_title=None, has_image=False):
        if not self.model:
            return 5.0
        features = self.extract_slide_features(slide_content, slide_title, has_image)
        feature_values = np.array([list(features.values())])
        score = self.model.predict(feature_values)[0]
        score = max(0, min(10, score))
        return score

class SlideStyleRecommender:
    def __init__(self):
        self.model_path = Path("models/style_recommender.joblib")
        self.feature_path = Path("models/style_features.joblib")
        self.model = None
        self.style_features = {
            'text_heavy': np.array([1.0, 0.2, 0.3, 0.1, 0.2]),
            'visual_focus': np.array([0.3, 1.0, 0.7, 0.5, 0.8]),
            'balanced': np.array([0.6, 0.6, 0.6, 0.5, 0.5]),
            'minimal': np.array([0.3, 0.3, 0.2, 0.1, 0.1]),
            'dynamic': np.array([0.7, 0.8, 1.0, 0.9, 0.7])
        }
        if self.model_path.exists() and self.feature_path.exists():
            try:
                self.model = joblib.load(self.model_path)
                self.style_features = joblib.load(self.feature_path)
                print(f"? Modèle de recommandation de style chargé")
            except Exception as e:
                print(f"?? Erreur chargement modèle de style: {e}")
                self.initialize_model()
        else:
            self.initialize_model()
    
    def initialize_model(self):
        try:
            styles = list(self.style_features.keys())
            features = np.array(list(self.style_features.values()))
            self.model = NearestNeighbors(n_neighbors=1)
            self.model.fit(features)
            print("? Modèle de recommandation de style initialisé")
        except Exception as e:
            print(f"?? Erreur initialisation modèle de style: {e}")
            self.model = None
    
    def extract_content_style_features(self, content, title=None, has_image=False):
        try:
            if not content or not isinstance(content, str):
                return np.array([0.5, 0.5, 0.5, 0.5, 0.5])
            features = np.zeros(5)
            clean_content = re.sub(r'!\[.*?\]\((.*?)\)', '', content)
            word_count = len(clean_content.split())
            features[0] = min(1.0, word_count / 500)
            image_count = len(re.findall(r'!\[.*?\]\(.*?\)', content))
            features[1] = min(1.0, (0.7 if has_image else 0) + (image_count * 0.15))
            bullet_points = len(re.findall(r'[-•*]', content))
            short_paragraphs = len(re.findall(r'\n\n.{10,100}\n\n', content))
            features[2] = min(1.0, (bullet_points * 0.05) + (short_paragraphs * 0.1))
            words = clean_content.split()
            if words:
                avg_word_length = np.mean([len(w) for w in words]) 
                features[3] = min(1.0, (avg_word_length - 3) / 5)
            else:
                features[3] = 0.0
            formal_keywords = ['conclusion', 'introduction', 'analyse', 'résultat', 'méthode']
            has_formal_keywords = any(keyword in clean_content.lower() for keyword in formal_keywords)
            has_formal_title = title and any(keyword in title.lower() for keyword in formal_keywords)
            features[4] = 0.3 + (0.3 if has_formal_keywords else 0) + (0.4 if has_formal_title else 0)
            return features
        except Exception as e:
            print(f"?? Erreur lors de l'extraction des caractéristiques de style: {e}")
            return np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    
    def recommend_style(self, content, title=None, has_image=False):
        try:
            if not self.model:
                return self.get_default_style(has_image)
            features = self.extract_content_style_features(content, title, has_image)
            distances, indices = self.model.kneighbors([features])
            styles = list(self.style_features.keys())
            if indices[0][0] < len(styles):
                recommended_style = styles[indices[0][0]]
            else:
                print("?? Index de style hors limites, utilisation du style par défaut")
                return self.get_default_style(has_image)
            style_params = self.generate_style_parameters(recommended_style, features, has_image)
            return style_params
        except Exception as e:
            print(f"?? Erreur lors de la recommandation de style: {e}")
            return self.get_default_style(has_image)
    
    def get_default_style(self, has_image):
        if has_image:
            return {
                'name': 'visual_focus',
                'text_size': 32,
                'animation_level': 0.7,
                'color_scheme': 'blue',
                'layout': 'image_right',
                'transition': 'fade'
            }
        else:
            return {
                'name': 'text_heavy',
                'text_size': 36,
                'animation_level': 0.3,
                'color_scheme': 'neutral',
                'layout': 'centered',
                'transition': 'fade'
            }
    
    def generate_style_parameters(self, style_name, features, has_image):
        base_params = {
            'text_heavy': {
                'text_size': 36,
                'animation_level': 0.3,
                'color_scheme': 'neutral',
                'layout': 'text_focus',
                'transition': 'fade'
            },
            'visual_focus': {
                'text_size': 32,
                'animation_level': 0.7,
                'color_scheme': 'vibrant',
                'layout': 'image_dominant',
                'transition': 'wipe_right'
            },
            'balanced': {
                'text_size': 34,
                'animation_level': 0.5,
                'color_scheme': 'balanced',
                'layout': 'split',
                'transition': 'fade'
            },
            'minimal': {
                'text_size': 30,
                'animation_level': 0.2,
                'color_scheme': 'monochrome',
                'layout': 'centered',
                'transition': 'fade'
            },
            'dynamic': {
                'text_size': 32,
                'animation_level': 0.9,
                'color_scheme': 'contrast',
                'layout': 'dynamic',
                'transition': 'zoom_in'
            }
        }
        if style_name in base_params:
            params = base_params[style_name].copy()
        else:
            print(f"?? Style '{style_name}' non reconnu, utilisation du style 'balanced'")
            params = base_params['balanced'].copy()
        params['name'] = style_name
        if 'transition' not in params:
            params['transition'] = 'fade'
        return params

class LayoutOptimizer:
    """
    Optimise la mise en page des diapositives en fonction du contenu
    """
    
    def __init__(self):
        self.model_path = Path("models/layout_optimizer.joblib")
        self.model = None
        if self.model_path.exists():
            try:
                self.model = joblib.load(self.model_path)
                print(f"? Modèle d'optimisation de mise en page chargé")
            except Exception as e:
                print(f"?? Erreur chargement modèle de mise en page: {e}")
        self.successful_layouts = {}
    
    def optimize_layout(self, content, title, images, style_params):
        layout_params = {}
        has_image = bool(images)
        clean_content = re.sub(r'!\[.*?\]\(.*?\)', '', content).strip()
        clean_content = re.sub(r'[#*_\s\n\r]', '', clean_content)
        is_image_only = not bool(clean_content) and has_image
        if is_image_only:
            print(f"?? Mode plein écran détecté - image sans texte")
            return {
                'text_x': 0.1, 'text_y': 0.85,
                'text_width': 0.8, 'text_height': 0.1,
                'image_x': 0.05, 'image_y': 0.05,
                'image_width': 0.9, 'image_height': 0.8,
                'title_position': 'bottom',
                'fullscreen_image': True
            }
        text_length = len(content)
        has_table = re.search(r'\|[-]+\|', content) is not None
        has_table = has_table or "|" in content and any(line.count('|') > 2 for line in content.split('\n'))
        large_images = []
        complex_images = []
        if has_image and images:
            for img_path in images:
                try:
                    if Path(img_path).exists():
                        img = Image.open(img_path)
                        width, height = img.size
                        aspect_ratio = width / height
                        if aspect_ratio > 1.8:
                            large_images.append(img_path)
                        if hasattr(self, '_has_grid_pattern') and self._has_grid_pattern(img.convert('L')):
                            complex_images.append(img_path)
                except Exception as e:
                    print(f"?? Erreur analyse image {img_path}: {e}")
        is_short_content = text_length < 500
        is_very_long_content = text_length > 1500
        layout_type = style_params.get('layout', 'balanced')
        if has_table:
            if has_image:
                return {
                    'text_x': 0.05, 'text_y': 0.15,
                    'text_width': 0.9, 'text_height': 0.45,
                    'image_x': 0.3, 'image_y': 0.65,
                    'image_width': 0.4, 'image_height': 0.3,
                    'title_position': 'top',
                    'text_size_factor': 0.9
                }
            else:
                return {
                    'text_x': 0.05, 'text_y': 0.15,
                    'text_width': 0.9, 'text_height': 0.75,
                    'title_position': 'top',
                    'text_size_factor': 0.9
                }
        if complex_images:
            return {
                'text_x': 0.05, 'text_y': 0.15,
                'text_width': 0.45, 'text_height': 0.75,
                'image_x': 0.55, 'image_y': 0.15,
                'image_width': 0.4, 'image_height': 0.7,
                'title_position': 'top',
                'text_size_factor': 0.95
            }
        if large_images:
            if is_short_content:
                return {
                    'text_x': 0.1, 'text_y': 0.55,
                    'text_width': 0.8, 'text_height': 0.4,
                    'image_x': 0.15, 'image_y': 0.15,
                    'image_width': 0.7, 'image_height': 0.35,
                    'title_position': 'top'
                }
            else:
                return {
                    'text_x': 0.1, 'text_y': 0.15,
                    'text_width': 0.8, 'text_height': 0.5,
                    'image_x': 0.15, 'image_y': 0.65,
                    'image_width': 0.7, 'image_height': 0.3,
                    'title_position': 'top'
                }
        if has_image:
            if layout_type == 'image_dominant':
                return {
                    'text_x': 0.1, 'text_y': 0.65,
                    'text_width': 0.8, 'text_height': 0.3,
                    'image_x': 0.1, 'image_y': 0.15,
                    'image_width': 0.8, 'image_height': 0.45,
                    'title_position': 'top'
                }
            elif layout_type == 'text_focus':
                return {
                    'text_x': 0.05, 'text_y': 0.15,
                    'text_width': 0.9, 'text_height': 0.5,
                    'image_x': 0.6, 'image_y': 0.68,
                    'image_width': 0.35, 'image_height': 0.28,
                    'title_position': 'top'
                }
            else:
                return {
                    'text_x': 0.05, 'text_y': 0.15,
                    'text_width': 0.45, 'text_height': 0.75,
                    'image_x': 0.55, 'image_y': 0.15,
                    'image_width': 0.4, 'image_height': 0.7,
                    'title_position': 'top'
                }
        else:
            return {
                'text_x': 0.1, 'text_y': 0.15,
                'text_width': 0.8, 'text_height': 0.7,
                'title_position': 'top',
                'text_size_factor': 1.0 if is_short_content else 0.9
            }
    
    def _has_grid_pattern(self, img_gray):
        try:
            width, height = img_gray.size
            row_samples = [img_gray.getpixel((i, height//2)) for i in range(0, width, 10)]
            col_samples = [img_gray.getpixel((width//2, i)) for i in range(0, height, 10)]
            threshold = 30
            row_edges = sum(1 for i in range(1, len(row_samples)) if abs(row_samples[i] - row_samples[i-1]) > threshold)
            col_edges = sum(1 for i in range(1, len(col_samples)) if abs(col_samples[i] - col_samples[i-1]) > threshold)
            return (row_edges > 3 and col_edges > 3)
        except Exception as e:
            print(f"?? Erreur détection grille: {e}")
            return False
