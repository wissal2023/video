# user_feedback.py
import json
import time
from pathlib import Path

class UserFeedbackSystem:
    """
    Système pour collecter et traiter les retours utilisateurs
    sur la qualité des présentations générées
    """
    
    def __init__(self):
        self.feedback_history = []
        self.session_ratings = {}
        self.feedback_file = Path("training_data/user_feedback.json")
        self.load_feedback_history()
        print("? Système de feedback utilisateur initialisé")
    
    def load_feedback_history(self):
        if self.feedback_file.exists():
            try:
                with open(self.feedback_file, 'r', encoding='utf-8') as f:
                    self.feedback_history = json.load(f)
                    print(f"?? {len(self.feedback_history)} feedbacks utilisateur chargés")
            except Exception as e:
                print(f"?? Erreur chargement historique feedback: {e}")
                self.feedback_history = []
    
    def save_feedback_history(self):
        try:
            with open(self.feedback_file, 'w', encoding='utf-8') as f:
                json.dump(self.feedback_history, f, ensure_ascii=False, indent=4)
            print(f"?? Historique de feedback sauvegardé ({len(self.feedback_history)} entrées)")
        except Exception as e:
            print(f"?? Erreur sauvegarde historique feedback: {e}")
    
    def add_feedback(self, slide_data, rating, comments=None):
        # Ajoute un feedback utilisateur à l'historique
        feedback = {
            "slide_data": slide_data,
            "rating": rating,
            "comments": comments,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.feedback_history.append(feedback)
        self.session_ratings[slide_data.get('title', 'Sans titre')] = rating
        self.save_feedback_history()
        return True
    
    def get_average_rating(self):
        if not self.session_ratings:
            return 0
        return sum(self.session_ratings.values()) / len(self.session_ratings)
    
    def get_recommendations(self):
        """
        Génère des recommandations d'amélioration basées sur les feedbacks
        
        Returns:
            recommendations: Liste de recommandations
        """
        recommendations = []
        
        # Analyser les notes par catégorie
        low_rated_slides = [title for title, rating in self.session_ratings.items() if rating < 5]
        
        if low_rated_slides:
            recommendations.append(f"Améliorer les diapositives suivantes: {', '.join(low_rated_slides)}")
        
        # Recommandations générales basées sur l'historique si disponible
        if len(self.feedback_history) > 5:
            avg_historical = sum(item['rating'] for item in self.feedback_history) / len(self.feedback_history)
            
            if avg_historical < 6:
                recommendations.append("Privilégier les présentations plus visuelles avec moins de texte")
            elif avg_historical < 8:
                recommendations.append("Continuer à améliorer l'équilibre texte/images")
        
        return recommendations
