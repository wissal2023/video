# deep_learning_models_fixed.py
# Module de Deep Learning pour le générateur de vidéos éducatives - Version corrigée

import os
import json
import numpy as np
import pickle
import time
from datetime import datetime

# Imports Deep Learning avec gestion d'erreur améliorée
DL_AVAILABLE = False
TfidfVectorizer = None
StandardScaler = None
tf = None

try:
    import tensorflow as tf
    print(tf.keras.__version__)
    from tensorflow.keras import layers, models
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler
    DL_AVAILABLE = True
    print("✅ TensorFlow et scikit-learn disponibles")
except ImportError as e:
    print(f"❌ Deep Learning non disponible: {e}")
    print("💡 Solutions possibles:")
    print("   1. pip install --upgrade tensorflow scikit-learn")
    print("   2. Installez Microsoft Visual C++ Redistributable")
    print("   3. Essayez: pip install tensorflow-cpu (version CPU uniquement)")
    print("   4. Ou utilisez conda: conda install tensorflow scikit-learn")
    
    # Classes de fallback pour éviter les erreurs
    class DummyVectorizer:
        def __init__(self, *args, **kwargs):
            pass
        def fit_transform(self, texts):
            return np.random.rand(len(texts), 100)  # Matrice aléatoire
        def transform(self, texts):
            return np.random.rand(len(texts), 100)
    
    class DummyScaler:
        def __init__(self):
            pass
        def fit_transform(self, data):
            return data
        def transform(self, data):
            return data
        def inverse_transform(self, data):
            return data
    
    TfidfVectorizer = DummyVectorizer
    StandardScaler = DummyScaler


class ContentClassifierDL:
    """Vrai modèle de Deep Learning pour classifier le contenu"""
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.label_encoder = {
            'technical': 0,
            'educational': 1, 
            'business': 2,
            'creative': 3
        }
        self.reverse_label = {v: k for k, v in self.label_encoder.items()}
        self.model_path = 'content_classifier_model'
        self.is_trained = False
        
        # Initialiser le vectorizer seulement si DL est disponible
        if DL_AVAILABLE:
            try:
                self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
            except Exception as e:
                print(f"Erreur initialisation vectorizer: {e}")
                self.vectorizer = None
        else:
            self.vectorizer = TfidfVectorizer()  # Dummy vectorizer
    
    def build_model(self, input_dim):
        """Construire le réseau de neurones"""
        if not DL_AVAILABLE or tf is None:
            return None
            
        try:
            model = models.Sequential([
                layers.Dense(512, activation='relu', input_shape=(input_dim,)),
                layers.Dropout(0.3),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(64, activation='relu'),
                layers.Dense(4, activation='softmax')  # 4 classes
            ])
            
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            return model
        except Exception as e:
            print(f"Erreur construction modèle: {e}")
            return None
    
    def prepare_training_data(self):
        """Préparer les données d'entraînement à partir de votre historique"""
        try:
            with open('video_generator_data.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            texts = []
            labels = []
            
            for file_info in data.get('processed_files', []):
                file_path = file_info.get('file_path', '')
                if os.path.exists(file_path):
                    try:
                        # Lire le contenu du fichier
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        # Analyser le contenu avec votre méthode existante
                        content_analysis = self.analyze_content_simple(content)
                        content_type = content_analysis.get('content_type', 'general')
                        
                        if content_type in self.label_encoder:
                            texts.append(content)
                            labels.append(self.label_encoder[content_type])
                    except Exception as e:
                        print(f"Erreur lecture fichier {file_path}: {e}")
                        continue
            
            # Si pas assez de données réelles, générer des données synthétiques
            if len(texts) < 20:
                print("📊 Génération de données synthétiques pour l'entraînement...")
                synthetic_texts, synthetic_labels = self.generate_synthetic_data()
                texts.extend(synthetic_texts)
                labels.extend(synthetic_labels)
            
            print(f"📊 Données d'entraînement: {len(texts)} échantillons")
            return texts, labels
            
        except Exception as e:
            print(f"Erreur préparation données: {e}")
            # Fallback : données synthétiques uniquement
            return self.generate_synthetic_data()
    
    def generate_synthetic_data(self):
        """Générer des données d'entraînement synthétiques"""
        synthetic_data = {
            'technical': [
                "API REST documentation with JSON endpoints and authentication methods for web development",
                "Machine learning algorithm implementation using Python TensorFlow and neural networks",
                "Database schema design with SQL queries optimization and performance tuning",
                "React components development with hooks state management and modern JavaScript",
                "Docker containerization and Kubernetes deployment for microservices architecture",
                "Cloud computing AWS services EC2 S3 Lambda serverless functions",
                "Data science analytics using pandas numpy matplotlib visualization libraries",
                "Backend development Node.js Express.js MongoDB database integration",
                "Frontend framework Vue.js Angular TypeScript progressive web applications",
                "DevOps CI/CD pipeline Jenkins GitHub Actions automated testing deployment"
            ],
            'educational': [
                "Introduction to programming concepts basic syntax variables functions loops",
                "Mathematics course covering algebra calculus differential equations statistics",
                "History lesson about world war events consequences social political changes",
                "Science tutorial explaining photosynthesis biology chemistry physics principles",
                "Language learning exercises grammar rules vocabulary pronunciation practice",
                "Economics fundamentals market analysis supply demand inflation monetary policy",
                "Psychology human behavior cognitive development learning memory processes",
                "Literature analysis poetry prose narrative techniques literary criticism",
                "Philosophy ethics logic reasoning critical thinking moral philosophy",
                "Geography climate change environmental science sustainable development"
            ],
            'business': [
                "Quarterly financial report revenue profit analysis market performance indicators",
                "Marketing strategy document new product launch customer acquisition campaigns",
                "Business plan comprehensive market research competitive analysis SWOT",
                "Sales presentation customer metrics conversion rates pipeline management",
                "Project management methodology agile scrum team organization workflow",
                "Human resources recruitment employee development performance evaluation",
                "Strategic planning corporate governance risk management compliance",
                "Financial analysis investment portfolio diversification asset allocation",
                "Operations management supply chain logistics inventory optimization",
                "Entrepreneurship startup funding venture capital business model innovation"
            ],
            'creative': [
                "Creative writing story characters plot development narrative structure dialogue",
                "Artistic design concepts visual composition color theory aesthetic principles",
                "Photography tutorial lighting composition techniques portrait landscape",
                "Music composition harmony rhythm melody songwriting production techniques",
                "Film production process script pre-production cinematography post-production",
                "Graphic design branding logo creation typography layout principles",
                "Web design user interface experience wireframing prototyping usability",
                "Animation 2D 3D motion graphics character design storytelling",
                "Fashion design textile patterns clothing construction trend analysis",
                "Architecture building design structural engineering sustainable construction"
            ]
        }
        
        texts = []
        labels = []
        
        # Générer plus de variabilité dans les données
        for content_type, examples in synthetic_data.items():
            for example in examples:
                # Version originale
                texts.append(example)
                labels.append(self.label_encoder[content_type])
                
                # Versions légèrement modifiées pour plus de diversité
                words = example.split()
                if len(words) > 5:
                    # Version raccourcie
                    short_version = ' '.join(words[:len(words)//2])
                    texts.append(short_version)
                    labels.append(self.label_encoder[content_type])
                    
                    # Version avec mots mélangés
                    if len(words) > 8:
                        mixed_words = words[:3] + words[5:8] + words[3:5]
                        mixed_version = ' '.join(mixed_words)
                        texts.append(mixed_version)
                        labels.append(self.label_encoder[content_type])
        
        print(f"📊 Données synthétiques générées: {len(texts)} échantillons")
        return texts, labels
    
    def train_model(self, save_path=None):
        """Entraîner le modèle de Deep Learning"""
        if not DL_AVAILABLE:
            print("❌ TensorFlow non disponible - impossible d'entraîner")
            return None, 0
            
        if save_path is None:
            save_path = self.model_path
            
        print("🧠 Entraînement du modèle de Deep Learning...")
        start_time = time.time()
        
        try:
            # Préparer les données
            texts, labels = self.prepare_training_data()
            
            if len(texts) == 0:
                print("❌ Aucune donnée d'entraînement disponible")
                return None, 0
            
            # Vectorisation
            print("🔄 Vectorisation des textes...")
            X = self.vectorizer.fit_transform(texts).toarray()
            y = np.array(labels)
            
            # Division train/test
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print(f"📊 Entraînement: {X_train.shape[0]} échantillons, Test: {X_test.shape[0]} échantillons")
            
            # Construire et entraîner le modèle
            self.model = self.build_model(X.shape[1])
            
            if self.model is None:
                print("❌ Impossible de construire le modèle")
                return None, 0
            
            # Entraînement avec callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    patience=15, 
                    restore_best_weights=True,
                    monitor='val_accuracy'
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    factor=0.5, 
                    patience=7,
                    monitor='val_loss'
                )
            ]
            
            print("🚀 Début de l'entraînement...")
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=100,
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )
            
            # Évaluation
            test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
            training_time = time.time() - start_time
            
            print(f"✅ Entraînement terminé en {training_time:.1f}s")
            print(f"🎯 Précision du modèle: {test_accuracy:.3f}")
            
            # Sauvegarder
            try:
                self.model.save(f'{save_path}.h5')
                with open(f'{save_path}_vectorizer.pkl', 'wb') as f:
                    pickle.dump(self.vectorizer, f)
                
                # Sauvegarder les métadonnées
                metadata = {
                    'accuracy': float(test_accuracy),
                    'training_time': training_time,
                    'training_samples': len(texts),
                    'created': datetime.now().isoformat()
                }
                with open(f'{save_path}_metadata.json', 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                self.is_trained = True
                print(f"💾 Modèle sauvegardé: {save_path}")
                
            except Exception as e:
                print(f"❌ Erreur sauvegarde: {e}")
            
            return history, test_accuracy
            
        except Exception as e:
            print(f"❌ Erreur durant l'entraînement: {e}")
            return None, 0
    
    def load_model(self, model_path=None):
        """Charger le modèle pré-entraîné"""
        if not DL_AVAILABLE:
            return False
            
        if model_path is None:
            model_path = self.model_path
            
        try:
            # Charger le modèle
            self.model = tf.keras.models.load_model(f'{model_path}.h5')
            
            # Charger le vectorizer
            with open(f'{model_path}_vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            # Charger les métadonnées si disponibles
            try:
                with open(f'{model_path}_metadata.json', 'r') as f:
                    metadata = json.load(f)
                    print(f"✅ Modèle chargé (précision: {metadata.get('accuracy', 'N/A'):.3f})")
            except:
                print("✅ Modèle Deep Learning chargé")
            
            self.is_trained = True
            return True
            
        except Exception as e:
            print(f"❌ Erreur chargement modèle: {e}")
            return False
    
    def predict_content_type(self, text):
        """Prédire le type de contenu avec Deep Learning"""
        if not DL_AVAILABLE or self.model is None:
            if not self.load_model():
                print("🔄 Fallback vers analyse heuristique...")
                return self.analyze_content_simple(text)
                
        if self.model is None:
            # Fallback vers analyse simple
            return self.analyze_content_simple(text)
        
        try:
            # Vectoriser le texte
            text_vector = self.vectorizer.transform([text]).toarray()
            
            # Prédiction
            predictions = self.model.predict(text_vector, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            
            predicted_type = self.reverse_label[predicted_class]
            
            return {
                'content_type': predicted_type,
                'confidence': confidence,
                'all_probabilities': {
                    self.reverse_label[i]: float(predictions[0][i]) 
                    for i in range(len(predictions[0]))
                },
                'method': 'deep_learning'
            }
            
        except Exception as e:
            print(f"❌ Erreur prédiction DL: {e}")
            return self.analyze_content_simple(text)
    
    def analyze_content_simple(self, content):
        """Version simple pour bootstrap les données ou fallback"""
        if isinstance(content, dict):
            # Si on reçoit déjà un résultat d'analyse, le retourner
            return content
            
        content_lower = content.lower()
        
        technical_keywords = [
            'api', 'algorithm', 'function', 'class', 'method', 'database', 
            'server', 'code', 'python', 'javascript', 'sql', 'framework',
            'tensorflow', 'machine learning', 'neural network', 'data'
        ]
        educational_keywords = [
            'learn', 'study', 'course', 'lesson', 'chapter', 'exercise', 
            'tutorial', 'education', 'teaching', 'student', 'academic'
        ]
        business_keywords = [
            'revenue', 'profit', 'market', 'sales', 'business', 'customer', 
            'strategy', 'analysis', 'financial', 'management', 'corporate'
        ]
        creative_keywords = [
            'story', 'design', 'creative', 'art', 'brand', 'marketing', 
            'narrative', 'visual', 'aesthetic', 'composition'
        ]
        
        scores = {
            'technical': sum(1 for kw in technical_keywords if kw in content_lower),
            'educational': sum(1 for kw in educational_keywords if kw in content_lower),
            'business': sum(1 for kw in business_keywords if kw in content_lower),
            'creative': sum(1 for kw in creative_keywords if kw in content_lower)
        }
        
        content_type = max(scores, key=scores.get) if max(scores.values()) > 0 else 'general'
        max_score = max(scores.values())
        
        # Calculer une confiance approximative
        total_keywords = sum(scores.values())
        confidence = max_score / max(total_keywords, 1) if total_keywords > 0 else 0.5
        
        return {
            'content_type': content_type,
            'confidence': min(confidence, 0.9),  # Limiter la confiance pour la méthode simple
            'method': 'heuristic'
        }


class PerformancePredictorDL:
    """Prédire les temps de génération avec Deep Learning"""
    
    def __init__(self):
        self.model = None
        self.scaler_X = StandardScaler() if DL_AVAILABLE else None
        self.scaler_y = StandardScaler() if DL_AVAILABLE else None
        self.model_path = 'performance_predictor_model'
        self.model_encoder = {
            'microsoft/phi-2': 0,
            'mistralai/Mistral-7B-v0.1': 1,
            'meta-llama/Llama-2-7b-hf': 2,
            'google/gemma-7b': 3,
            'google/flan-t5-xxl': 4,
            'microsoft/phi-1_5': 5,
            'bigscience/bloom-7b1': 6
        }
    
    def build_performance_model(self, input_dim):
        """Construire un modèle de régression pour prédire les performances"""
        if not DL_AVAILABLE:
            return None
            
        try:
            model = models.Sequential([
                layers.Dense(128, activation='relu', input_shape=(input_dim,)),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.2),
                layers.Dense(32, activation='relu'),
                layers.Dense(16, activation='relu'),
                layers.Dense(1, activation='linear')  # Temps de génération
            ])
            
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            return model
        except Exception as e:
            print(f"Erreur construction modèle performance: {e}")
            return None
    
    def predict_processing_time(self, file_path, model_name, content_analysis=None):
        """Prédire le temps de traitement pour un fichier et un modèle"""
        if not DL_AVAILABLE or self.model is None:
            # Fallback vers estimation heuristique
            return self._estimate_time_heuristic(file_path, model_name, content_analysis)
        
        try:
            # Extraire les features
            features = self._extract_features(file_path, model_name, content_analysis)
            features_scaled = self.scaler_X.transform([features])
            
            # Prédiction
            prediction = self.model.predict(features_scaled, verbose=0)[0][0]
            prediction_unscaled = self.scaler_y.inverse_transform([[prediction]])[0][0]
            
            return max(60, int(prediction_unscaled))  # Minimum 1 minute
            
        except Exception as e:
            print(f"Erreur prédiction performance: {e}")
            return self._estimate_time_heuristic(file_path, model_name, content_analysis)
    
    def _extract_features(self, file_path, model_name, content_analysis):
        """Extraire des features pour la prédiction"""
        features = []
        
        # Features du fichier
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        features.append(file_size / (1024 * 1024))  # Taille en MB
        
        # Features du modèle
        model_id = self.model_encoder.get(model_name, 0)
        features.append(model_id)
        
        # Features du contenu
        if content_analysis:
            complexity_score = {'simple': 1, 'medium': 2, 'complex': 3}.get(
                content_analysis.get('complexity', 'medium'), 2
            )
            features.append(complexity_score)
            
            estimated_length = content_analysis.get('estimated_length', 500)
            features.append(estimated_length / 1000)  # En milliers de mots
            
            content_type_score = {
                'technical': 3, 'educational': 2, 'business': 1, 'creative': 2
            }.get(content_analysis.get('content_type', 'general'), 1)
            features.append(content_type_score)
        else:
            features.extend([2, 0.5, 1])  # Valeurs par défaut
        
        return features
    
    def _estimate_time_heuristic(self, file_path, model_name, content_analysis):
        """Estimation heuristique du temps de traitement"""
        base_times = {
            'microsoft/phi-1_5': 180,      # 3 min
            'microsoft/phi-2': 240,        # 4 min
            'google/gemma-7b': 300,        # 5 min
            'mistralai/Mistral-7B-v0.1': 360,  # 6 min
            'meta-llama/Llama-2-7b-hf': 420,   # 7 min
            'google/flan-t5-xxl': 480,     # 8 min
            'bigscience/bloom-7b1': 420    # 7 min
        }
        
        base_time = base_times.get(model_name, 300)
        
        # Ajustements selon le contenu
        if content_analysis:
            if content_analysis.get('complexity') == 'complex':
                base_time *= 1.5
            elif content_analysis.get('complexity') == 'simple':
                base_time *= 0.8
                
            if content_analysis.get('content_type') == 'technical':
                base_time *= 1.2
        
        # Ajustement selon la taille du fichier
        if os.path.exists(file_path):
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if file_size_mb > 5:
                base_time *= 1.3
            elif file_size_mb < 1:
                base_time *= 0.9
        
        return int(base_time)


# Classe principale d'intégration
class DeepLearningManagerEnhanced:
    """Gestionnaire Deep Learning amélioré pour l'interface PyQt"""
    
    def __init__(self, data_file="video_generator_data.json"):
        self.data_file = data_file
        self.learning_data = self.load_data()
        
        # Initialiser les modèles DL avec gestion d'erreur
        try:
            self.content_classifier = ContentClassifierDL()
            self.performance_predictor = PerformancePredictorDL()
        except Exception as e:
            print(f"Erreur initialisation modèles: {e}")
            self.content_classifier = None
            self.performance_predictor = None
        
        # Essayer de charger les modèles existants
        self.dl_available = DL_AVAILABLE
        if self.dl_available and self.content_classifier:
            print("🧠 Initialisation du Deep Learning...")
            if not self.content_classifier.load_model():
                print("📚 Modèle non trouvé - sera entraîné au premier usage")
        else:
            print("⚠️ Deep Learning non disponible - Mode heuristique")
    
    def load_data(self):
        """Charge les données sauvegardées"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"Données chargées: {len(data.get('processed_files', []))} fichiers traités")
                    return data
        except Exception as e:
            print(f"Erreur lors du chargement des données: {e}")
        
        return {
            "processed_files": [],
            "model_performance": {},
            "deep_learning_enabled": True,
            "improvement_level": 1,
            "total_videos_generated": 0,
            "user_preferences": {},
            "learning_patterns": [],
            "fusion_mode_enabled": True
        }
    
    def save_data(self):
        """Sauvegarde les données"""
        try:
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(self.learning_data, f, ensure_ascii=False, indent=2)
            print("Données sauvegardées avec succès")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde: {e}")
    
    def analyze_file_content(self, file_path):
        """Analyse avec Deep Learning ou fallback heuristique"""
        try:
            # Lire le contenu du fichier
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Tenter l'analyse avec Deep Learning
            if (self.dl_available and 
                self.content_classifier and 
                self.learning_data.get("deep_learning_enabled", True)):
                
                dl_result = self.content_classifier.predict_content_type(content)
                
                # Ajouter des informations complémentaires
                dl_result.update({
                    'complexity': self._determine_complexity(content),
                    'estimated_length': len(content.split()),
                    'file_size': os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                    'file_type': os.path.splitext(file_path)[1].lower()
                })
                
                return dl_result
            else:
                # Fallback vers analyse heuristique
                return self._analyze_content_heuristic(file_path)
                
        except Exception as e:
            print(f"Erreur analyse fichier: {e}")
            return self._analyze_content_heuristic(file_path)
    
    def _determine_complexity(self, content):
        """Détermine la complexité du contenu"""
        word_count = len(content.split())
        
        # Compter les éléments complexes
        complex_indicators = [
            content.count('```'),  # Blocs de code
            content.count('$$'),   # Formules mathématiques
            content.count('http'), # URLs
            len([w for w in content.split() if len(w) > 10])  # Mots longs
        ]
        
        complexity_score = sum(complex_indicators)
        
        if word_count > 2000 or complexity_score > 10:
            return "complex"
        elif word_count > 500 or complexity_score > 3:
            return "medium"
        else:
            return "simple"
    
    def _analyze_content_heuristic(self, file_path):
        """Analyse heuristique de fallback"""
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            
            content_info = {
                "complexity": "simple",
                "content_type": "general",
                "estimated_length": 0,
                "method": "heuristic",
                "confidence": 0.6
            }
            
            if file_ext in ['.md', '.txt']:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    sample_text = f.read()[:2000]
                
                if sample_text and self.content_classifier:
                    result = self.content_classifier.analyze_content_simple(sample_text)
                    content_info.update(result)
                    content_info["estimated_length"] = len(sample_text.split())
                    content_info["complexity"] = self._determine_complexity(sample_text)
            
            # Analyse basée sur le nom du fichier pour PDF/DOCX
            file_name = os.path.basename(file_path).lower()
            name_keywords = {
                'technical': ['tech', 'api', 'documentation', 'manual', 'guide'],
                'educational': ['course', 'lesson', 'tutorial', 'education', 'formation'],
                'business': ['business', 'report', 'analysis', 'presentation', 'meeting'],
                'creative': ['creative', 'story', 'design', 'marketing', 'brand']
            }
            
            for content_type, keywords in name_keywords.items():
                if any(word in file_name for word in keywords):
                    content_info["content_type"] = content_type
                    break
            
            # Complexité basée sur la taille
            if file_size > 5 * 1024 * 1024:  # > 5MB
                content_info["complexity"] = "complex"
            elif file_size > 1 * 1024 * 1024:  # > 1MB
                content_info["complexity"] = "medium"
            
            return content_info
            
        except Exception as e:
            print(f"Erreur analyse heuristique: {e}")
            return {
                "complexity": "simple",
                "content_type": "general",
                "estimated_length": 0,
                "method": "fallback",
                "confidence": 0.5
            }
    
    def get_specialized_models(self, file_type, content_info):
        """Retourne les modèles spécialisés selon le type de fichier et le contenu"""
        complexity = content_info.get("complexity", "medium")
        content_type = content_info.get("content_type", "general")
        
        # Logique de recommandation identique à votre version originale
        recommendations = []
        
        if file_type == '.pdf':
            if content_type == "technical":
                if complexity == "complex":
                    recommendations = ["google/flan-t5-xxl", "google/gemma-7b", "mistralai/Mistral-7B-v0.1"]
                else:
                    recommendations = ["google/gemma-7b", "microsoft/phi-2", "google/flan-t5-xxl"]
            elif content_type == "educational":
                recommendations = ["mistralai/Mistral-7B-v0.1", "google/flan-t5-xxl", "microsoft/phi-2"]
            elif content_type == "business":
                recommendations = ["microsoft/phi-2", "google/gemma-7b", "mistralai/Mistral-7B-v0.1"]
            elif content_type == "creative":
                recommendations = ["meta-llama/Llama-2-7b-hf", "mistralai/Mistral-7B-v0.1", "microsoft/phi-2"]
            else:  # general
                if complexity == "complex":
                    recommendations = ["google/flan-t5-xxl", "microsoft/phi-2", "google/gemma-7b"]
                else:
                    recommendations = ["microsoft/phi-2", "google/flan-t5-xxl", "google/gemma-7b"]
                    
        elif file_type == '.md':
            if content_type == "technical":
                recommendations = ["google/gemma-7b", "microsoft/phi-2", "google/flan-t5-xxl"]
            elif content_type == "educational":
                recommendations = ["mistralai/Mistral-7B-v0.1", "microsoft/phi-2", "google/flan-t5-xxl"]
            else:
                recommendations = ["microsoft/phi-2", "mistralai/Mistral-7B-v0.1", "google/flan-t5-xxl"]
                
        elif file_type == '.docx':
            if content_type == "creative":
                recommendations = ["meta-llama/Llama-2-7b-hf", "mistralai/Mistral-7B-v0.1", "bigscience/bloom-7b1"]
            elif content_type == "business":
                recommendations = ["mistralai/Mistral-7B-v0.1", "microsoft/phi-2", "google/gemma-7b"]
            elif content_type == "educational":
                recommendations = ["mistralai/Mistral-7B-v0.1", "google/flan-t5-xxl", "microsoft/phi-2"]
            else:
                recommendations = ["mistralai/Mistral-7B-v0.1", "microsoft/phi-2", "google/flan-t5-xxl"]
                
        else:  # Autres types
            recommendations = ["microsoft/phi-2", "mistralai/Mistral-7B-v0.1", "google/flan-t5-xxl"]
        
        return recommendations[:3]
    
    def get_recommendations_for_file(self, file_path):
        """Obtient des recommandations basées sur l'analyse DL + historique"""
        file_type = os.path.splitext(file_path)[1]
        
        recommendations = {
            "suggested_models": [],
            "optimization_params": {},
            "estimated_time": 300,
            "reasoning": "",
            "confidence": 0.7
        }
        
        # Analyser le contenu avec DL
        content_info = self.analyze_file_content(file_path)
        
        # Obtenir les recommandations spécialisées
        specialized_models = self.get_specialized_models(file_type, content_info)
        
        # Estimer le temps avec DL si disponible
        if self.dl_available and self.performance_predictor and specialized_models:
            estimated_time = self.performance_predictor.predict_processing_time(
                file_path, specialized_models[0], content_info
            )
            recommendations["estimated_time"] = estimated_time
        
        recommendations["suggested_models"] = specialized_models
        recommendations["confidence"] = content_info.get("confidence", 0.7)
        
        # Construire le raisonnement
        method = content_info.get("method", "heuristic")
        content_type = content_info.get("content_type", "general")
        complexity = content_info.get("complexity", "medium")
        
        if method == "deep_learning":
            recommendations["reasoning"] = f"Deep Learning: {content_type} ({complexity}) - Confiance: {recommendations['confidence']:.2f}"
        else:
            recommendations["reasoning"] = f"Analyse heuristique: {content_type} ({complexity})"
        
        return recommendations
    
    def add_processed_file(self, file_path, models_used, video_paths, processing_time):
        """Ajoute un fichier traité à l'historique"""
        file_info = {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "models_used": models_used,
            "video_paths": video_paths,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat(),
            "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else 0,
            "file_type": os.path.splitext(file_path)[1]
        }
        
        self.learning_data["processed_files"].append(file_info)
        self.learning_data["total_videos_generated"] += len(video_paths)
        self.update_improvement_level()
        self.save_data()
        
        # Ré-entraîner le modèle périodiquement
        if (self.dl_available and 
            self.content_classifier and
            len(self.learning_data["processed_files"]) % 10 == 0 and 
            len(self.learning_data["processed_files"]) > 20):
            print("🔄 Ré-entraînement du modèle avec nouvelles données...")
            try:
                self.content_classifier.train_model()
            except Exception as e:
                print(f"Erreur ré-entraînement: {e}")
    
    def update_improvement_level(self):
        """Met à jour le niveau d'amélioration basé sur l'expérience"""
        total_files = len(self.learning_data["processed_files"])
        if total_files >= 50:
            self.learning_data["improvement_level"] = 5
        elif total_files >= 25:
            self.learning_data["improvement_level"] = 4
        elif total_files >= 10:
            self.learning_data["improvement_level"] = 3
        elif total_files >= 5:
            self.learning_data["improvement_level"] = 2
        else:
            self.learning_data["improvement_level"] = 1
    
    def get_deep_learning_params(self):
        """Retourne les paramètres d'amélioration deep learning"""
        level = self.learning_data["improvement_level"]
        
        params = {
            "enhancement_level": level,
            "quality_boost": min(level * 0.2, 1.0),
            "processing_optimization": True,
            "advanced_features": level >= 3,
            "auto_enhancement": level >= 4,
            "smart_rendering": level >= 5,
            "dl_available": self.dl_available,
            "model_trained": self.content_classifier.is_trained if (self.dl_available and self.content_classifier) else False
        }
        
        return params


# Auto-initialisation si exécuté directement
if __name__ == "__main__":
    print("🧠 Test du module Deep Learning...")
    
    if DL_AVAILABLE:
        try:
            classifier = ContentClassifierDL()
            
            # Test avec du texte
            test_text = "This is a technical documentation about API development using Python and TensorFlow for machine learning applications."
            result = classifier.predict_content_type(test_text)
            
            print(f"Texte de test: {test_text[:50]}...")
            print(f"Prédiction: {result}")
        except Exception as e:
            print(f"Erreur durant le test: {e}")
    else:
        print("❌ TensorFlow non disponible")
        print("🔄 Test en mode heuristique...")
        try:
            classifier = ContentClassifierDL()
            test_text = "This is a technical documentation about API development using Python."
            result = classifier.analyze_content_simple(test_text)
            print(f"Résultat heuristique: {result}")
        except Exception as e:
            print(f"Erreur test heuristique: {e}")