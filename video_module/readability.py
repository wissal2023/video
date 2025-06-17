# readability.py
import re

class ReadabilityEvaluator:
    """
    Évalue et améliore la lisibilité du contenu des diapositives
    """
    
    def __init__(self):
        self.model_path = "models/readability_model.joblib"
        self.model = None
        # Essayer de charger un modèle existant
        if Path(self.model_path).exists():
            try:
                self.model = joblib.load(self.model_path)
                print(f"? Modèle d'évaluation de lisibilité chargé")
            except Exception as e:
                print(f"?? Erreur chargement modèle de lisibilité: {e}")  
    
    def evaluate_readability(self, text):
        """
        Évalue la lisibilité du texte et retourne un score
        
        Args:
            text: Texte à évaluer
            
        Returns:
            score: Score de lisibilité (0-10)
            issues: Liste de problèmes identifiés
            suggestions: Suggestions d'amélioration
        """
        score = 5  # Score par défaut
        issues = []
        suggestions = []
        
        if not text or not text.strip():
            return score, issues, suggestions
        
        # Nettoyage de base
        clean_text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
        clean_text = re.sub(r'#{1,6}\s+', '', clean_text)
        
        # Métriques simples
        words = clean_text.split()
        word_count = len(words)
        sentences = re.split(r'[.!?]+', clean_text)
        sentence_count = sum(1 for s in sentences if s.strip())
        
        avg_words_per_sentence = word_count / max(1, sentence_count)
        avg_word_length = sum(len(word) for word in words) / max(1, word_count)
        
        # Vérification de problèmes courants
        if word_count > 200:
            issues.append("Texte trop long")
            suggestions.append("Réduire le texte à moins de 200 mots")
            score -= 1
        
        if avg_words_per_sentence > 25:
            issues.append("Phrases trop longues")
            suggestions.append("Raccourcir les phrases à moins de 25 mots")
            score -= 1
        
        if avg_word_length > 7:
            issues.append("Mots complexes")
            suggestions.append("Utiliser des mots plus simples")
            score -= 0.5
        
        # Vérification de la structure
        bullet_points = len(re.findall(r'[-•*]', clean_text))
        if word_count > 100 and bullet_points < 3:
            issues.append("Manque de structure")
            suggestions.append("Ajouter des listes à puces pour structurer le contenu")
            score -= 1
        
        # Vérification de la répétition
        word_freq = {}
        for word in words:
            word = word.lower()
            if len(word) > 3:  # Ignorer les mots courts
                word_freq[word] = word_freq.get(word, 0) + 1
        
        repeated_words = [word for word, freq in word_freq.items() if freq > 3]
        if repeated_words:
            issues.append(f"Répétition de mots: {', '.join(repeated_words[:3])}")
            suggestions.append("Varier le vocabulaire pour éviter les répétitions")
            score -= min(len(repeated_words) * 0.2, 1)
        
        # Ajustement final du score
        score = max(1, min(10, score + 5))  # Garantir entre 1 et 10
        
        return score, issues, suggestions
    
    def improve_readability(self, text):
        """
        Améliore automatiquement la lisibilité du texte
        
        Args:
            text: Texte à améliorer
            
        Returns:
            improved_text: Texte amélioré
            changes: Liste des changements effectués
        """
        if not text or not text.strip():
            return text, []
        
        changes = []
        improved_text = text
        
        # 1. Diviser les phrases trop longues
        def split_long_sentences(match):
            sentence = match.group(0)
            if len(sentence.split()) > 25:
                words = sentence.split()
                midpoint = len(words) // 2
                # Trouver un point de coupure naturel (virgule, etc.)
                for i in range(midpoint-3, midpoint+3):
                    if 0 <= i < len(words) and any(p in words[i] for p in [',', ';', ':', '-']):
                        midpoint = i + 1
                        break
                
                first_half = ' '.join(words[:midpoint])
                second_half = ' '.join(words[midpoint:])
                
                # Capitaliser la première lettre de la seconde moitié
                if second_half:
                    second_half = second_half[0].upper() + second_half[1:]
                
                changes.append("Phrase longue divisée")
                return f"{first_half}. {second_half}"
            return sentence
        
        # Appliquer la division de phrases
        sentence_pattern = re.compile(r'[^.!?]+[.!?]')
        improved_text = sentence_pattern.sub(split_long_sentences, improved_text)
        
        # 2. Ajouter des puces pour les listes potentielles
        paragraphs = improved_text.split('\n\n')
        for i, para in enumerate(paragraphs):
            # Détecter les paragraphes qui sont des listes potentielles
            if len(para.split()) > 30 and ',' in para and ';' in para:
                # Convertir en liste à puces
                items = re.split(r'[;,] (?=[A-Z])', para)
                if len(items) >= 3:  # Au moins 3 éléments
                    bullet_list = '\n\n'
                    for item in items:
                        item = item.strip()
                        if item:
                            if not item.endswith('.'):
                                item += '.'
                            bullet_list += f"* {item}\n"
                    
                    paragraphs[i] = bullet_list
                    changes.append("Paragraphe converti en liste à puces")
        
        improved_text = '\n\n'.join(paragraphs)
        
        # 3. Mettre en gras les termes importants
        if '**' not in improved_text:  # Ne pas ajouter si déjà du gras
            for title_word in re.findall(r'##\s+(.*?)$', text, re.MULTILINE):
                for word in title_word.split():
                    if len(word) > 4:
                        # Mettre en gras la première occurrence du mot-clé
                        pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
                        match = pattern.search(improved_text)
                        if match and '**' not in improved_text[max(0, match.start()-2):match.end()+2]:
                            improved_text = improved_text[:match.start()] + '**' + match.group(0) + '**' + improved_text[match.end():]
                            changes.append(f"Mot-clé mis en évidence: {word}")
        
        return improved_text, changes
    