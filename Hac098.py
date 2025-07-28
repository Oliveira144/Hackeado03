import streamlit as st
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional
import math


class MarkovModel:
    """
    Modelo simples de cadeia de Markov de ordem 1 para transições entre estados ('V', 'C').
    Ajuda a estimar a probabilidade condicional do próximo evento dado o atual.
    """
    def __init__(self):
        self.transitions: Dict[str, Counter] = defaultdict(Counter)

    def train(self, sequence: List[str]):
        for i in range(len(sequence) - 1):
            self.transitions[sequence[i]][sequence[i + 1]] += 1

    def predict_next_prob(self, current: str) -> Dict[str, float]:
        counts = self.transitions.get(current, {})
        total = sum(counts.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in counts.items()}


class CasinoAnalyzer:
    def __init__(self, history: List[str]):
        self.results = history

    def analyze_micro_patterns(self) -> List[Dict[str, Any]]:
        """
        Analisa os últimos resultados para detectar padrões micro de repetição dupla (2x2)
        e alternâncias artificiais que indicam manipulação.
        """
        patterns = []
        if len(self.results) < 6:
            return patterns

        last6 = self.results[-6:]
        double_pattern_count = sum(
            1 for i in range(0, 6, 2)
            if i + 1 < 6 and last6[i] == last6[i + 1] and last6[i] != 'E'
        )

        if double_pattern_count >= 2:
            patterns.append({
                'type': 'micro_double_pattern',
                'strength': double_pattern_count / 3,
                'risk': 'critical' if double_pattern_count == 3 else 'high',
                'description': f'Padrão 2x2 repetitivo ({double_pattern_count}/3)',
                'manipulation': 'CRÍTICA - Sistema forçando padrão' if double_pattern_count == 3 else 'ALTA',
                'predictability': 85
            })

        last8_non_empate = [r for r in self.results if r != 'E'][-8:]
        if len(last8_non_empate) >= 6:
            micro_alternations = sum(
                1 for i in range(1, min(6, len(last8_non_empate)))
                if last8_non_empate[i] != last8_non_empate[i - 1]
            )
            if micro_alternations >= 4:
                patterns.append({
                    'type': 'micro_alternation',
                    'strength': micro_alternations / 5,
                    'risk': 'critical' if micro_alternations == 5 else 'high',
                    'description': f'Micro-alternação suspeita ({micro_alternations}/5)',
                    'manipulation': 'Sistema induzindo alternância artificial',
                    'predictability': 90
                })
        return patterns

    def detect_hidden_cycles(self) -> List[Dict[str, Any]]:
        patterns = []
        non_empate = [r for r in self.results if r != 'E']
        if len(non_empate) < 12:
            return patterns

        for cycle_size in range(3, 7):
            cycles = [''.join(non_empate[i:i + cycle_size]) for i in range(len(non_empate) - cycle_size + 1)]
            cycle_counts = Counter(cycles)
            repeated = [(cycle, count) for cycle, count in cycle_counts.items() if count >= 2]
            if repeated:
                most_repeated, count = max(repeated, key=lambda x: x[1])
                patterns.append({
                    'type': 'hidden_cycle',
                    'cycle_size': cycle_size,
                    'pattern': most_repeated,
                    'repetitions': count,
                    'strength': min(count / 3, 1),
                    'risk': 'high' if count >= 3 else 'medium',
                    'description': f'Ciclo oculto detectado: "{most_repeated}" ({count}x)',
                    'manipulation': 'Sistema usando ciclo programado' if count >= 3 else 'Possível ciclo induzido',
                    'predictability': 70 + (count * 5)
                })
        return patterns

    def analyze_compensation_patterns(self) -> List[Dict[str, Any]]:
        patterns = []
        non_empate = [r for r in self.results if r != 'E']
        n = len(non_empate)
        if n < 20:
            return patterns

        windows = [12, 15, 18]

        for window_size in windows:
            if n >= window_size:
                window = non_empate[-window_size:]
                c_count = window.count('C')
                v_count = window.count('V')
                imbalance = abs(c_count - v_count)
                balance_ratio = imbalance / window_size

                if balance_ratio < 0.1 and window_size >= 15:
                    patterns.append({
                        'type': 'artificial_balance',
                        'window_size': window_size,
                        'balance': f"{c_count}C/{v_count}V",
                        'strength': 1 - balance_ratio,
                        'risk': 'high',
                        'description': f'Equilíbrio artificial em {window_size} jogadas',
                        'manipulation': 'Sistema forçando distribuição 50/50',
                        'predictability': 85
                    })

                if balance_ratio > 0.4:
                    underrepresented = 'C' if c_count < v_count else 'V'
                    patterns.append({
                        'type': 'compensation_pending',
                        'window_size': window_size,
                        'imbalance': imbalance,
                        'favored_color': underrepresented,
                        'strength': balance_ratio,
                        'risk': 'medium',
                        'description': f'Compensação pendente: {c_count}C vs {v_count}V',
                        'manipulation': f'Sistema deve favorecer {underrepresented}',
                        'predictability': 60 + (balance_ratio * 20)
                    })
        return patterns

    def analyze_strategic_ties(self) -> List[Dict[str, Any]]:
        # Placeholder, pode ser customizado para empates
        return []

    def shannon_entropy(self, data: List[str]) -> float:
        total = len(data)
        if total == 0:
            return 0.0
        counter = Counter(data)
        return -sum((count / total) * math.log2(count / total) for count in counter.values())

    def analyze_entropy(self, window=12, low_threshold=1.0) -> List[Dict[str, Any]]:
        non_empate = [r for r in self.results if r != 'E']
        if len(non_empate) < window:
            return []
        window_data = non_empate[-window:]
        entropy = self.shannon_entropy(window_data)
        patterns = []
        if entropy < low_threshold:
            patterns.append({
                'type': 'low_entropy',
                'entropy': entropy,
                'risk': 'high',
                'description': f'Entropia baixa ({entropy:.2f}) nas últimas {window} jogadas (padrão previsível)'
            })
        return patterns

    def detect_regime_change(self, window=15) -> List[Dict[str, Any]]:
        non_empate = [r for r in self.results if r != 'E']
        patterns = []
        if len(non_empate) < 2 * window:
            return patterns
        early = non_empate[-2 * window:-window]
        late = non_empate[-window:]
        freq_early = Counter(early)
        freq_late = Counter(late)
        total_early = len(early)
        total_late = len(late)
        diff_C = abs((freq_early.get('C', 0) / total_early) - (freq_late.get('C', 0) / total_late))
        diff_V = abs((freq_early.get('V', 0) / total_early) - (freq_late.get('V', 0) / total_late))
        if max(diff_C, diff_V) > 0.3:
            patterns.append({
                'type': 'regime_change',
                'risk': 'high',
                'description': f'Mudança brusca de padrão em {window} jogos (ΔC:{diff_C:.2f}, ΔV:{diff_V:.2f})'
            })
        return patterns

    def detect_near_cycles(self, cycle_size=3, max_misses=1) -> List[Dict[str, Any]]:
        non_empate = [r for r in self.results if r != 'E']
        patterns = []
        if len(non_empate) < cycle_size * 2:
            return patterns
        segments = [''.join(non_empate[i:i + cycle_size]) for i in range(len(non_empate) - cycle_size + 1)]
        for i, seg_a in enumerate(segments):
            for j, seg_b in enumerate(segments):
                if i >= j:
                    continue
                misses = sum(a != b for a, b in zip(seg_a, seg_b))
                if 0 < misses <= max_misses:
                    patterns.append({
                        'type': 'near_cycle',
                        'pattern': seg_a,
                        'similar_to': seg_b,
                        'misses': misses,
                        'cycle_size': cycle_size,
                        'risk': 'high' if misses == 1 else 'medium',
                        'description': f'Quase-ciclo: "{seg_a}" ~ "{seg_b}" ({misses} divergência)'
                    })
        return patterns

    def assess_risk(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        risk_score = 0
        risk_factors = []

        for pattern in patterns:
            p_type = pattern['type']
            strength = pattern.get('strength', 0)
            if p_type == 'micro_double_pattern' and strength >= 0.8:
                risk_score += 70
                risk_factors.append('🚨 Padrão 2x2 crítico detectado')
            elif p_type == 'micro_alternation' and strength >= 0.8:
                risk_score += 65
                risk_factors.append('⚠️ Alternação artificial crítica')
            elif p_type == 'hidden_cycle' and pattern.get('repetitions', 0) >= 3:
                risk_score += 60
                risk_factors.append(f'🔄 Ciclo programado ativo ({pattern.get("repetitions")}x)')
            elif p_type == 'artificial_balance':
                risk_score += 55
                risk_factors.append('⚖️ Equilíbrio artificial forçado')
            elif p_type == 'intentional_break':
                risk_score += 50
                risk_factors.append('💥 Quebra intencional detectada')
            elif p_type.startswith('strategic_tie'):
                risk_score += 40
                risk_factors.append('🔶 Empate estratégico detectado')
            elif p_type == 'near_cycle':
                risk_score += 65
                risk_factors.append(f'🌀 Quase-ciclo detectado: {pattern["description"]}')
            elif p_type == 'low_entropy':
                risk_score += 70
                risk_factors.append(f'📉 Entropia baixa: sistema muito previsível ({pattern["entropy"]:.2f})')
            elif p_type == 'regime_change':
                risk_score += 60
                risk_factors.append(f'⚡ Mudança brusca de padrão detectada')

        if risk_score >= 80:
            level = 'critical'
        elif risk_score >= 55:
            level = 'high'
        elif risk_score >= 30:
            level = 'medium'
        else:
            level = 'low'

        return {'level': level, 'score': min(risk_score, 100), 'factors': risk_factors}

    def detect_manipulation(self, patterns: List[Dict[str, Any]], risk: Dict[str, Any]) -> Dict[str, Any]:
        manipulation_score = 0
        manipulation_signs = []

        for pattern in patterns:
            predictability = pattern.get('predictability', 0)
            if predictability >= 90:
                manipulation_score += 80
                manipulation_signs.append(f"🤖 Padrão altamente artificial: {pattern['description']}")
            elif predictability >= 80:
                manipulation_score += 60
                manipulation_signs.append(f"🎯 Padrão programado: {pattern['description']}")
            elif predictability >= 70:
                manipulation_score += 40
                manipulation_signs.append(f"⚙️ Padrão suspeito: {pattern['description']}")

        if any(p['type'] == 'near_cycle' for p in patterns) and risk['score'] >= 60:
            manipulation_score = max(manipulation_score, 70)
            manipulation_signs.append("🚨 Indícios fortes de manipulação camuflada (quase-ciclos detectados)")

        if any(p['type'] == 'low_entropy' for p in patterns) and risk['score'] >= 60:
            manipulation_score = max(manipulation_score, 75)
            manipulation_signs.append("🚨 Sistema altamente previsível detectado (baixa entropia)")

        if manipulation_score >= 80:
            level = 'critical'
        elif manipulation_score >= 60:
            level = 'high'
        elif manipulation_score >= 35:
            level = 'medium'
        else:
            level = 'low'

        return {'level': level, 'score': min(manipulation_score, 100), 'signs': manipulation_signs}

    def build_markov_model(self) -> Optional[MarkovModel]:
        non_empate = [r for r in self.results if r != 'E']
        if len(non_empate) < 20:
            return None
        mm = MarkovModel()
        mm.train(non_empate)
        return mm

    def evaluate_markov_prediction(self, mm: MarkovModel) -> Dict[str, Any]:
        """
        Avalia a probabilidade do próximo evento segundo a cadeia de Markov.
        Retorna padrão de risco e sugestão baseado na probabilidade.
        """
        non_empate = [r for r in self.results if r != 'E']
        if not non_empate or mm is None:
            return {}

        last = non_empate[-1]
        probs = mm.predict_next_prob(last)
        if not probs:
            return {}

        # Encontra cor com maior probabilidade
        most_prob_color = max(probs, key=probs.get)
        max_prob = probs[most_prob_color]

        # Define risco por baixa probabilidade máxima (baixa certeza)
        if max_prob < 0.4:
            risk = 'high'
            description = f'Modelo Markov: próxima jogada incerta, probabilidade máxima {max_prob:.2f}'
        else:
            risk = 'low'
            description = f'Modelo Markov: próxima jogada mais provável é "{most_prob_color}" com probabilidade {max_prob:.2f}'

        return {
            'type': 'markov_prediction',
            'predicted_color': most_prob_color,
            'probability': max_prob,
            'risk': risk,
            'description': description
        }

    def make_prediction(self, patterns: List[Dict[str, Any]], risk: Dict[str, Any], manipulation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combina heurística e modelo Markov para predição final, equilibrando confiança e riscos.
        """
        prediction = {'color': None, 'confidence': 0, 'reasoning': '', 'strategy': 'AGUARDAR MELHORES CONDIÇÕES'}

        # Parada imediata se risco critico/manipulação critica
        if risk['level'] == 'critical' or manipulation['level'] == 'critical':
            prediction.update({
                'reasoning': '🚨 CONDIÇÕES CRÍTICAS - Manipulação máxima detectada',
                'strategy': 'PARAR COMPLETAMENTE'
            })
            return prediction

        # Evitar apostas em alto nível de manipulação
        if manipulation['level'] == 'high':
            prediction.update({
                'reasoning': '⛔ Manipulação alta - Evitar apostas',
                'strategy': 'AGUARDAR NORMALIZAÇÃO'
            })
            return prediction

        # Usa Markov para previsão probabilística adicional
        mm = self.build_markov_model()
        markov_pred = self.evaluate_markov_prediction(mm) if mm else {}

        # Ajusta decisão com base nas análises heurísticas + markov
        compensation_pattern = next((p for p in patterns if p['type'] == 'compensation_pending'), None)
        cycle_pattern = next((p for p in patterns if p['type'] == 'hidden_cycle' and p.get('repetitions', 0) >= 2), None)

        # Caso de compensação e baixo risco
        if compensation_pattern and risk['level'] == 'low' and manipulation['level'] == 'low':
            color = compensation_pattern['favored_color']
            confidence = min(75, 55 + (compensation_pattern['strength'] * 20))
            prediction.update({
                'color': color,
                'confidence': confidence,
                'reasoning': f'Compensação estatística esperada: {compensation_pattern["description"]}',
                'strategy': 'APOSTAR COMPENSAÇÃO'
            })
            return prediction

        # Seguir ciclo se conferido e risco baixo
        if cycle_pattern and risk['level'] == 'low' and manipulation['level'] == 'low':
            next_color = self.predict_next_in_cycle(cycle_pattern['pattern'])
            if next_color:
                prediction.update({
                    'color': next_color,
                    'confidence': min(70, 50 + (cycle_pattern['repetitions'] * 5)),
                    'reasoning': f'Ciclo detectado: "{cycle_pattern["pattern"]}" ({cycle_pattern["repetitions"]}x)',
                    'strategy': 'SEGUIR CICLO'
                })
                return prediction

        # Se Markov tem alta probabilidade para um próximo evento
        if markov_pred:
            prob = markov_pred['probability']
            color = markov_pred['predicted_color']
            risk_markov = markov_pred['risk']

            if prob >= 0.5 and risk_markov == 'low' and risk['level'] in ['low', 'medium'] and manipulation['level'] == 'low':
                confidence = min(65, prob * 100)
                prediction.update({
                    'color': color,
                    'confidence': confidence,
                    'reasoning': f'Predição baseada em cadeia de Markov: {markov_pred["description"]}',
                    'strategy': 'APOSTAR BASEADO EM MODELO ESTOCÁSTICO'
                })
                return prediction

        # Fallback para aposta na cor mais frequente no histórico se nada acima
        non_empate = [r for r in self.results if r != 'E']
        if not non_empate:
            return prediction

        counter = Counter(non_empate)
        most_common_color, count = counter.most_common(1)[0]
        confidence = (count / len(non_empate)) * 100

        prediction.update({
            'color': most_common_color,
            'confidence': min(confidence, 75),
            'reasoning': f'Aposta baseada em frequência histórica de "{most_common_color}" ({count}/{len(non_empate)})',
            'strategy': 'APOSTAR NA PRINCIPAL COR'
        })
        return prediction

    def predict_next_in_cycle(self, pattern: str) -> Optional[str]:
        non_empate = [r for r in self.results if r != 'E']
        if not pattern or not non_empate:
            return None
        cycle_len = len(pattern)
        for i in range(len(non_empate)):
            if non_empate[i] != pattern[i % cycle_len]:
                return None
        pos = len(non_empate) % cycle_len
        return pattern[pos]


def main():
    st.title("Casino Analyzer")

    if 'history' not in st.session_state:
        st.session_state.history = []

    col1, col2, col3 = st.columns(3)

    # Botões com emojis para inserir resultados
    if col1.button("🔴"):
        st.session_state.history.append('V')  # Casa = vermelho
    if col2.button("🔵"):
        st.session_state.history.append('C')  # Visitante = azul
    if col3.button("🟡"):
        st.session_state.history.append('E')  # Empate = amarelo

    col_clear, col_undo = st.columns(2)
    if col_clear.button("Limpar Histórico"):
        st.session_state.history = []
    if col_undo.button("Apagar Último Resultado"):
        if st.session_state.history:
            st.session_state.history.pop()

    if st.session_state.history:
        st.write("### Histórico Atual (Mais recente à esquerda):")
        color_map = {'V': '🔴', 'C': '🔵', 'E': '🟡'}
        history_display = ''.join(color_map.get(r, '⬜') for r in reversed(st.session_state.history))
        st.markdown(f"**{history_display}**")
        st.write("*Nota: Histórico mostrado do resultado mais recente (esquerda) ao mais antigo (direita).")
    else:
        st.info("Use os botões acima para inserir resultados e iniciar análise.")
        return

    analyzer = 
