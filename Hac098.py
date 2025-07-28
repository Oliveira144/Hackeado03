import streamlit as st
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional
import math

# Parâmetros de janela: foco total entre últimos 10 e 27 resultados
WINDOW_MIN = 10
WINDOW_MAX = 27

class MarkovModel:
    """Cadeia de Markov de ordem até 2 para previsão condicional."""
    def __init__(self, order=2):
        self.order = order
        self.transitions: Dict[Any, Counter] = defaultdict(Counter)

    def train(self, sequence: List[str]):
        if len(sequence) < self.order + 1:
            return
        for i in range(len(sequence) - self.order):
            key = tuple(sequence[i:i+self.order])
            self.transitions[key][sequence[i+self.order]] += 1

    def predict_next_prob(self, context: List[str]) -> Dict[str, float]:
        if len(context) != self.order:
            return {}
        counts = self.transitions.get(tuple(context), {})
        total = sum(counts.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in counts.items()}

class CasinoAnalyzer:
    def __init__(self, results: List[str]):
        self.results = results

    def _extract_window(self) -> List[str]:
        if len(self.results) >= WINDOW_MAX:
            window = self.results[-WINDOW_MAX:]
        else:
            window = self.results[:]
        if len(window) < WINDOW_MIN:
            return []
        return window[-WINDOW_MIN:]

    def analyze_patterns(self) -> List[Dict[str, Any]]:
        patterns = []
        window = self._extract_window()
        if not window:
            return patterns
        # Micro-padrão 2x2
        if len(window) >= 6:
            last6 = window[-6:]
            double_count = sum(1 for i in range(0, 6, 2)
                               if i+1 < 6 and last6[i] == last6[i+1] and last6[i] != 'E')
            if double_count >= 2:
                risk_level = 'crítico' if double_count == 3 else 'alto'
                patterns.append({
                    'type': 'micro_double_pattern',
                    'desc': f'Padrão 2x2 repetitivo ({double_count}/3 em {last6})',
                    'risk': risk_level
                })
        # Alta alternância
        if len(window) >= 8:
            alt_count = sum(1 for i in range(1, 8)
                            if window[-i] != window[-i-1] and window[-i] != 'E' and window[-i-1] != 'E')
            if alt_count >= 4:
                risk_level = 'crítico' if alt_count == 7 else 'alto'
                patterns.append({'type': 'micro_alternation', 'desc': f'Alternância alta ({alt_count}/7)', 'risk': risk_level})
        # Compensação
        c = window.count('C')
        v = window.count('V')
        diff = abs(c - v)
        if len(window) >= WINDOW_MIN:
            if diff <= 1:
                patterns.append({'type': 'artificial_balance', 'desc': f'Equilíbrio estatístico ({c}C x {v}V)', 'risk': 'suspeito'})
            if diff >= int(0.45 * len(window)):
                favored = 'C' if c < v else 'V'
                patterns.append({'type': 'compensation_pending', 'desc': f'Compensação pendente na cor {favored}', 'risk': 'alto'})
        # Entropia baixa
        ent = self.shannon_entropy([x for x in window if x in ['C','V']])
        if ent < 0.7:
            patterns.append({'type': 'low_entropy', 'desc': f'Entropia baixa: {ent:.2f}', 'risk': 'crítico'})
        # Ciclos e quase-ciclos
        for size in [3, 4, 5]:
            if len(window) < 2 * size:
                continue
            segments = [''.join(window[i:i+size]) for i in range(len(window)-size+1)]
            count_segments = Counter(segments)
            most_common, count = count_segments.most_common(1)[0]
            if count >= 2:
                risk_level = 'alto' if count == 2 else 'crítico'
                patterns.append({'type': 'hidden_cycle', 'desc': f'Ciclo quase-repetido: "{most_common}" ({count}x)', 'risk': risk_level})
        return patterns

    def shannon_entropy(self, seq: List[str]) -> float:
        total = len(seq)
        if total == 0:
            return 0.0
        freq = Counter(seq)
        return -sum((c/total)*math.log2(c/total) for c in freq.values())

    def risk_and_signal(self, patterns: List[Dict[str, Any]]) -> str:
        risk_map = {'crítico': 3, 'alto': 2, 'suspeito': 1}
        score = sum(risk_map.get(p['risk'], 0) for p in patterns)
        if score >= 5:
            return "crítico"
        elif score >= 3:
            return "alto"
        elif score >= 1:
            return "moderado"
        return "baixo"

    def build_markov_model(self, order=2) -> Optional[MarkovModel]:
        eventos = [r for r in self.results if r in ['C', 'V']]
        if len(eventos) < order+1:
            return None
        mm = MarkovModel(order=order)
        mm.train(eventos)
        return mm

    def markov_predict(self) -> Dict[str, Any]:
        window = self._extract_window()
        eventos = [r for r in window if r in ['C', 'V']]
        if len(eventos) < max(WINDOW_MIN, 6):
            return {'color': None, 'conf': 0, 'support': 'Histórico insuficiente.'}
        # Modelo ordem 2 preferencial
        mk2 = self.build_markov_model(order=2)
        if mk2 and len(eventos) >= 2:
            probs = mk2.predict_next_prob(eventos[-2:])
            if probs:
                cor, prob = max(probs.items(), key=lambda t: t[1])
                if prob >= 0.60:
                    return {'color': cor, 'conf': prob*100, 'support': f'Cadeia Markov(2): {probs}'}
        # Fallback ordem 1
        mk1 = self.build_markov_model(order=1)
        if mk1 and len(eventos) >= 1:
            probs = mk1.predict_next_prob(eventos[-1:])
            if probs:
                cor, prob = max(probs.items(), key=lambda t: t[1])
                if prob >= 0.60:
                    return {'color': cor, 'conf': prob*100, 'support': f'Cadeia Markov(1): {probs}'}
        # Fallback frequência simples
        freq = Counter(eventos)
        cor, q = freq.most_common(1)[0]
        return {'color': cor, 'conf': (q/len(eventos))*100, 'support': 'Maior frequência na janela.'}

def main():
    st.title("CasinoAnalyzer PRO - Análise Avançada de Manipulação e Predição")

    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'predictions_log' not in st.session_state:
        st.session_state.predictions_log = []
    if 'accuracy_log' not in st.session_state:
        st.session_state.accuracy_log = []

    col1, col2, col3 = st.columns(3)
    if col1.button("🔴 (Casa - Vermelho)"):
        st.session_state.history.append('V')
    if col2.button("🔵 (Visitante - Azul)"):
        st.session_state.history.append('C')
    if col3.button("🟡 (Empate - Amarelo)"):
        st.session_state.history.append('E')

    col_clear, col_undo = st.columns(2)
    if col_clear.button("Limpar Histórico"):
        st.session_state.history = []
        st.session_state.predictions_log = []
        st.session_state.accuracy_log = []
    if col_undo.button("Apagar Último Resultado"):
        if st.session_state.history:
            st.session_state.history.pop()
        if st.session_state.predictions_log:
            st.session_state.predictions_log.pop()
        if st.session_state.accuracy_log:
            st.session_state.accuracy_log.pop()

    if not st.session_state.history:
        st.info("Use os botões acima para inserir os resultados do jogo e iniciar a análise.")
        return

    # Exibição do histórico
    st.subheader("Histórico atual (mais recente à esquerda):")
    color_map = {'V': '🔴', 'C': '🔵', 'E': '🟡'}
    hist_disp = ''.join(color_map.get(r, '⬜') for r in reversed(st.session_state.history))
    st.markdown(f"**{hist_disp}**")

    analyzer = CasinoAnalyzer(st.session_state.history)
    patterns = analyzer.analyze_patterns()
    risk = analyzer.risk_and_signal(patterns)
    markov_pred = analyzer.markov_predict()

    # Conferência das predições com resultados reais
    eventos = [r for r in st.session_state.history if r in ['C', 'V']]
    idx_pred = len(st.session_state.predictions_log)
    if len(eventos) >= 2 and len(st.session_state.predictions_log) > 0:
        pred_idx = len(st.session_state.predictions_log) - 1
        real_idx = pred_idx + 1
        if real_idx < len(eventos):
            prev_pred = st.session_state.predictions_log[pred_idx]
            real_result = eventos[real_idx]
            if prev_pred.get('color') is not None:
                if len(st.session_state.accuracy_log) < len(st.session_state.predictions_log):
                    st.session_state.accuracy_log.append(prev_pred.get('color') == real_result)

    if len(st.session_state.predictions_log) < len(eventos):
        st.session_state.predictions_log.append(markov_pred)

    # Exibir avaliação de risco
    st.markdown("## Avaliação de Risco 🚦")
    st.markdown(f"- Nível de risco da janela {WINDOW_MIN}-{WINDOW_MAX}: **{risk}**")
    with st.expander("Padrões detectados e níveis de risco"):
        if patterns:
            for p in patterns:
                st.write(f"- [{p['risk'].upper()}] {p['desc']}")
        else:
            st.write("Nenhum padrão relevante detectado na janela atual.")

    # Predição e BOTÃO interativo para entrada
    st.header("Predição do Próximo Resultado")
    if risk == "crítico":
        st.error("🚨 Manipulação crítica detectada! Sistema em pausa para proteção.\n"
                 "Aguarde a alimentação de mais dados para retomada automática das análises e sinais.")
    else:
        color = markov_pred.get('color')
        conf = markov_pred.get('conf', 0)
        support = markov_pred.get('support', '')
        if color:
            emoji = {'V': '🔴', 'C': '🔵'}.get(color, color)
            st.success(f"**Sinal sugerido:** {emoji}  (Confiança: {conf:.1f}%)")
            st.write(f"Base analítica: {support}")
            # Botão explícito para entrada
            if st.button(f"Apostar {emoji}"):
                st.write(f"✅ Entrada registrada para a cor {emoji}. Boa sorte!")
        else:
            st.info("Ainda sem confiança suficiente para sugerir sinal no momento.")

    # Painel de performance
    st.markdown("---")
    st.markdown("## Performance do Sistema (conferência automática)")
    if st.session_state.accuracy_log:
        total = len(st.session_state.accuracy_log)
        acertos = sum(st.session_state.accuracy_log)
        taxa = (acertos / total) * 100
        st.markdown(f"- Total de sinais avaliados: {total}")
        st.markdown(f"- Acertos: {acertos}")
        st.markdown(f"- Taxa de acerto: **{taxa:.2f}%**")
        with st.expander("Últimas 20 Conferências (Acertos/Erros)"):
            ultimas = st.session_state.accuracy_log[-20:]
            start = total - len(ultimas) + 1
            for i, acerto in enumerate(ultimas, start=start):
                st.write(f"#{i}: {'✅ Acertou' if acerto else '❌ Errou'}")
    else:
        st.write("Ainda sem dados suficientes para avaliar desempenho.")

if __name__ == "__main__":
    main()
