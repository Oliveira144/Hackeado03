import streamlit as st
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional
import math

# Parâmetros para janela de análise (foco nos últimos 10 a 27 resultados)
WINDOW_MIN = 10
WINDOW_MAX = 27

class MarkovModel:
    """Cadeia de Markov de ordem variável (até 3) para previsão condicional."""
    def __init__(self, order=3):
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
        total = len(self.results)
        if total < WINDOW_MIN:
            return []
        elif WINDOW_MIN <= total <= WINDOW_MAX:
            # Prioriza analisar toda a janela de resultados entre 10 e 27 (todos resultados disponíveis até 27)
            return self.results[:]
        else:
            # Para mais de 27 resultados, considera sempre os últimos 27 resultados
            return self.results[-WINDOW_MAX:]

    def analyze_patterns(self) -> List[Dict[str, Any]]:
        patterns = []
        window = self._extract_window()
        if not window:
            return patterns

        # Micro-padrão 2x2 repetitivo
        if len(window) >= 6:
            last6 = window[-6:]
            # Considera apenas C ou V para este padrão, excluindo E
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
            # Conta alternâncias entre C e V, ignorando E
            alt_count = sum(1 for i in range(1, 8)
                            if window[-i] != window[-i-1] and window[-i] != 'E' and window[-i-1] != 'E')
            if alt_count >= 4:
                risk_level = 'crítico' if alt_count == 7 else 'alto'
                patterns.append({
                    'type': 'micro_alternation',
                    'desc': f'Alternância alta ({alt_count}/7)',
                    'risk': risk_level
                })

        # Compensação (balanceamento / pendência) - apenas C e V
        cv_window = [x for x in window if x in ['C', 'V']]
        if len(cv_window) >= WINDOW_MIN:
            c = cv_window.count('C')
            v = cv_window.count('V')
            total_cv = len(cv_window)
            diff = abs(c - v)

            # Ajuste para balanço artificial: se a diferença for muito pequena para o tamanho da janela
            # E se a contagem de C e V é significativa
            if total_cv >= 10 and diff <= 1:
                patterns.append({
                    'type': 'artificial_balance',
                    'desc': f'Equilíbrio estatístico artificial ({c}C x {v}V)',
                    'risk': 'suspeito'
                })
            
            # Ajuste para compensação pendente: se a diferença for significativa em relação ao total
            if total_cv >= 10 and diff >= int(0.40 * total_cv): # Limiar de 40% da janela
                favored = 'C' if c < v else 'V'
                patterns.append({
                    'type': 'compensation_pending',
                    'desc': f'Compensação pendente na cor {favored} (diferença {diff})',
                    'risk': 'alto'
                })

        # Entropia baixa (baixa aleatoriedade) - apenas C e V
        ent = self.shannon_entropy([x for x in window if x in ['C','V']])
        if ent < 0.7 and len([x for x in window if x in ['C','V']]) >= WINDOW_MIN: # Garante dados suficientes
            patterns.append({
                'type': 'low_entropy',
                'desc': f'Entropia baixa: {ent:.2f}',
                'risk': 'crítico'
            })

        # Ciclos e quase-ciclos - considera C, V e E para detectar repetições gerais
        for size in [3, 4, 5]:
            if len(window) < 2 * size:
                continue
            segments = [''.join(window[i:i+size]) for i in range(len(window)-size+1)]
            counter_segs = Counter(segments)
            for most_common_seg, count in counter_segs.most_common(1): # Pega o mais comum
                if count >= 2 and len(most_common_seg) == size: # Garante que é um ciclo repetido
                    risk_level = 'alto' if count == 2 else 'crítico'
                    patterns.append({
                        'type': 'hidden_cycle',
                        'desc': f'Ciclo quase-repetido: "{most_common_seg}" ({count}x)',
                        'risk': risk_level
                    })

        return patterns

    def shannon_entropy(self, seq: List[str]) -> float:
        total = len(seq)
        if total == 0:
            return 0.0
        freq = Counter(seq)
        # Handle cases where log2(0) might occur due to rounding for very small counts if not careful
        # Using a small epsilon to avoid log(0) for robustness if needed, but Counter ensures counts are > 0
        return -sum((c/total)*math.log2(c/total) for c in freq.values() if c > 0)

    def risk_and_signal(self, patterns: List[Dict[str, Any]]) -> str:
        risk_map = {'crítico': 3, 'alto': 2, 'suspeito': 1}
        score = sum(risk_map.get(p['risk'], 0) for p in patterns)
        
        # Ajuste dos limiares para classificação do risco geral
        if score >= 5: # Um ou mais padrões críticos ou múltiplos altos
            return "crítico"
        elif score >= 3: # Um padrão alto ou múltiplos suspeitos/mistos
            return "alto"
        elif score >= 1: # Pelo menos um padrão suspeito
            return "moderado"
        else:
            return "baixo"

    def build_markov_model(self, order=3) -> Optional[MarkovModel]:
        eventos = [r for r in self.results if r in ['C', 'V']]
        if len(eventos) < order + 1:
            return None
        mm = MarkovModel(order=order)
        mm.train(eventos)
        return mm

    def get_dynamic_confidence_threshold(self) -> float:
        """Calcula um limiar de confiança dinâmico baseado na acurácia histórica."""
        if not st.session_state.accuracy_log:
            return 0.65 # Limiar padrão inicial, ligeiramente mais conservador
        
        total_predictions = len(st.session_state.accuracy_log)
        if total_predictions < 5: # Precisa de um mínimo de dados para ajustar
            return 0.65
            
        current_accuracy = sum(st.session_state.accuracy_log) / total_predictions
        
        # Ajusta o limiar:
        # Se a acurácia é muito boa, pode-se ser um pouco menos rígido.
        # Se a acurácia é baixa, ser mais rígido.
        if current_accuracy >= 0.75:
            return 0.55
        elif current_accuracy >= 0.65:
            return 0.60
        else: # Abaixo de 65% de acurácia, exige mais confiança
            return 0.70
    
    def markov_predict_adaptive(self) -> Dict[str, Any]:
        """
        Realiza a predição Markov adaptativa, combinando resultados de diferentes ordens
        e usando um limiar de confiança dinâmico.
        """
        window = self._extract_window()
        eventos = [r for r in window if r in ['C', 'V']] # Foco em C e V para Markov
        
        if len(eventos) < 6: # Mínimo para alguma análise de padrão e Markov básica
            return {'color': None, 'conf': 0, 'support': 'Histórico insuficiente para previsão robusta.'}

        all_preds_info = []

        # Tenta predições com diferentes ordens de Markov
        # Ordem 3
        if len(eventos) >= 3:
            mk3 = self.build_markov_model(order=3)
            if mk3:
                probs = mk3.predict_next_prob(eventos[-3:])
                if probs:
                    cor, prob = max(probs.items(), key=lambda t: t[1])
                    all_preds_info.append({'color': cor, 'prob': prob, 'order': 3, 'context': eventos[-3:]})

        # Ordem 2
        if len(eventos) >= 2:
            mk2 = self.build_markov_model(order=2)
            if mk2:
                probs = mk2.predict_next_prob(eventos[-2:])
                if probs:
                    cor, prob = max(probs.items(), key=lambda t: t[1])
                    all_preds_info.append({'color': cor, 'prob': prob, 'order': 2, 'context': eventos[-2:]})

        # Ordem 1
        if len(eventos) >= 1:
            mk1 = self.build_markov_model(order=1)
            if mk1:
                probs = mk1.predict_next_prob(eventos[-1:])
                if probs:
                    cor, prob = max(probs.items(), key=lambda t: t[1])
                    all_preds_info.append({'color': cor, 'prob': prob, 'order': 1, 'context': eventos[-1:]})

        # Combinação das previsões de Markov
        if all_preds_info:
            combined_scores = defaultdict(float)
            support_details = []

            for pred_info in all_preds_info:
                # Pondera a previsão: maior ordem e maior probabilidade têm mais peso
                # Adiciona um pequeno valor para evitar divisão por zero se prob for 0
                weight = pred_info['order'] * (pred_info['prob'] + 0.01) 
                combined_scores[pred_info['color']] += weight
                support_details.append(f"Mk({pred_info['order']}) ctxt '{''.join(pred_info['context'])}'->'{pred_info['color']}' ({pred_info['prob']:.2f})")

            # Encontra a cor com a maior pontuação combinada
            if combined_scores:
                final_color, max_combined_score = max(combined_scores.items(), key=lambda t: t[1])
                
                # Normaliza a pontuação combinada para uma "confiança" percentual
                # Maxima pontuação possível seria se todas as ordens concordassem com prob 1.0
                max_possible_score = sum(p['order'] * (1.0 + 0.01) for p in all_preds_info)
                conf_normalized = (max_combined_score / max_possible_score) if max_possible_score > 0 else 0

                dynamic_threshold = self.get_dynamic_confidence_threshold()
                
                if conf_normalized >= dynamic_threshold:
                    return {
                        'color': final_color,
                        'conf': conf_normalized * 100,
                        'support': f"Previsão combinada Markov: {' | '.join(support_details)}. Limiar: {dynamic_threshold:.2f}"
                    }
        
        # Fallback para frequência simples se Markov não gerar predição robusta
        freq_eventos = Counter(eventos)
        if freq_eventos:
            cor_freq, q_freq = freq_eventos.most_common(1)[0]
            conf_freq = (q_freq/len(eventos))*100
            if conf_freq >= 50: # Mínimo de 50% para frequência simples
                return {'color': cor_freq, 'conf': conf_freq, 'support': 'Maior frequência na janela recente.'}

        return {'color': None, 'conf': 0, 'support': 'Não foi possível gerar previsão robusta com dados atuais.'}


def main():
    st.set_page_config(page_title="CasinoAnalyzer PRO", layout="centered", initial_sidebar_state="expanded")
    st.title("CasinoAnalyzer PRO - Análise Avançada de Padrões e Predição")
    st.markdown("Analise o comportamento dos resultados e receba sugestões para suas apostas.")

    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'predictions_log' not in st.session_state:
        st.session_state.predictions_log = []
    if 'accuracy_log' not in st.session_state:
        st.session_state.accuracy_log = []
    if 'last_prediction_made' not in st.session_state: # Flag para controlar quando uma predição foi feita
        st.session_state.last_prediction_made = None

    # Botões para entrada de resultados
    st.subheader("Registrar Novo Resultado:")
    col1, col2, col3 = st.columns(3)
    if col1.button("🔴 Casa (Vermelho)", use_container_width=True):
        st.session_state.history.append('V')
        st.session_state.last_prediction_made = None # Resetar flag ao adicionar novo resultado
    if col2.button("🔵 Visitante (Azul)", use_container_width=True):
        st.session_state.history.append('C')
        st.session_state.last_prediction_made = None
    if col3.button("🟡 Empate (Amarelo)", use_container_width=True):
        st.session_state.history.append('E')
        st.session_state.last_prediction_made = None

    col_clear, col_undo = st.columns(2)
    if col_clear.button("Limpar Histórico", type="secondary", use_container_width=True):
        st.session_state.history = []
        st.session_state.predictions_log = []
        st.session_state.accuracy_log = []
        st.session_state.last_prediction_made = None
    if col_undo.button("Apagar Último Resultado", type="secondary", use_container_width=True):
        if st.session_state.history:
            st.session_state.history.pop()
        if st.session_state.predictions_log: # Tenta remover a última predição se houver
            st.session_state.predictions_log.pop()
        if st.session_state.accuracy_log: # Tenta remover a última acurácia se houver
            st.session_state.accuracy_log.pop()
        st.session_state.last_prediction_made = None

    if not st.session_state.history:
        st.info("Use os botões acima para inserir os resultados do jogo e iniciar a análise.")
        return

    # Mostrar histórico com emojis (mais recente à esquerda)
    st.subheader("Histórico de Resultados (mais recente à esquerda):")
    color_map = {'V': '🔴', 'C': '🔵', 'E': '🟡'}
    hist_disp = ''.join(color_map.get(r, '⬜') for r in reversed(st.session_state.history))
    st.markdown(f"**{hist_disp}**")

    # Executar análise
    analyzer = CasinoAnalyzer(st.session_state.history)
    patterns = analyzer.analyze_patterns()
    risk = analyzer.risk_and_signal(patterns)
    # A predição agora usa o método adaptativo
    markov_pred = analyzer.markov_predict_adaptive()

    # Conferência automática da última predição e resultado real
    # Ajusta a lógica para garantir que estamos comparando a predição feita ANTES do resultado atual
    eventos_cv = [r for r in st.session_state.history if r in ['C', 'V']]
    
    # Se há mais resultados C/V do que predições registradas,
    # e se a última predição registrada corresponde ao penúltimo resultado C/V
    # e o último resultado C/V é um novo resultado para ser conferido
    if len(eventos_cv) > 0 and len(st.session_state.predictions_log) > 0:
        # A última predição no log é para o evento que *acabou* de acontecer
        last_recorded_pred = st.session_state.predictions_log[-1]
        
        # O resultado real a ser conferido é o último evento C/V do histórico
        real_result_for_check = eventos_cv[-1] 
        
        # Para evitar dupla conferência e garantir que a predição foi para este resultado
        # Verifica se o log de acurácia já tem essa conferência ou se o resultado atual é novo
        if len(st.session_state.accuracy_log) < len(st.session_state.predictions_log):
            # A predição foi feita para o resultado que acaba de ser inserido
            # Só confere se a predição era válida e não crítica
            if last_recorded_pred.get('color') is not None and last_recorded_pred.get('risk_level') != 'crítico':
                acertou = (last_recorded_pred['color'] == real_result_for_check)
                st.session_state.accuracy_log.append(acertou)

    # Registrar predição para o *próximo* resultado, se ainda não foi registrada para o estado atual
    # Só registra se a predição for válida (cor diferente de None) E o risco não for "crítico"
    # E se a predição ainda não foi registrada para o estado atual do histórico
    if st.session_state.last_prediction_made is None: # Só tenta registrar se não registrou ainda
        if markov_pred.get('color') is not None and risk != 'crítico':
            pred_to_add = markov_pred.copy()
            pred_to_add['risk_level'] = risk # Anexa o nível de risco do momento da predição
            st.session_state.predictions_log.append(pred_to_add)
            st.session_state.last_prediction_made = pred_to_add # Marca que uma predição foi feita

    # Exibir avaliação de risco e padrões detectados
    st.markdown("---")
    st.markdown("## Avaliação de Risco 🚦")
    st.markdown(f"- Nível de risco da janela {WINDOW_MIN}-{WINDOW_MAX}: **{risk.upper()}**")
    with st.expander("Padrões detectados e níveis de risco (clique para expandir)"):
        if patterns:
            for p in patterns:
                st.write(f"- [{p['risk'].upper()}] {p['desc']}")
        else:
            st.write("Nenhum padrão relevante detectado na janela atual.")

    # Predição e sugestão de aposta
    st.markdown("---")
    st.header("Predição do Próximo Resultado")
    if risk == "crítico":
        st.error(
            "🚨 **ALERTA: Manipulação crítica detectada!**\n"
            "O sistema recomenda **NÃO APOSTAR** no momento, para sua proteção.\n"
            "Aguarde mais resultados para que a análise possa se reajustar."
        )
    else:
        color = markov_pred.get('color')
        conf = markov_pred.get('conf', 0)
        support = markov_pred.get('support', '')
        emoji_map = {'V': '🔴', 'C': '🔵'}
        emoji = emoji_map.get(color, None)

        if emoji and conf > 0: # Confiança maior que zero
            st.subheader(f"Sinal para o Próximo Resultado:")
            if conf >= analyzer.get_dynamic_confidence_threshold():
                st.success(f"**SINAL FORTE:** Apostar {emoji}  (Confiança: {conf:.1f}%)")
                st.write(f"Base analítica: {support}")
                if st.button(f"Confirmar Aposta em {emoji}"):
                    st.success(f"✅ Sua intenção de aposta em {emoji} foi registrada. Boa sorte!")
                    # Aqui você poderia adicionar lógica de integração real ou logging
            elif conf > 50: # Entre 50% e o limiar dinâmico
                st.warning(f"**SINAL MODERADO:** Considerar {emoji} com cautela (Confiança: {conf:.1f}%)")
                st.write(f"Base analítica: {support}")
            else: # Abaixo de 50%, mas ainda com alguma indicação
                st.info(f"**INDICAÇÃO LEVE:** {emoji} é a tendência mais provável (Confiança: {conf:.1f}%)")
                st.write(f"Base analítica: {support}")
        else:
            st.info("Ainda sem confiança suficiente para sugerir uma aposta no momento.")
            st.write(f"Detalhes: {support}")


    # Painel de performance automática
    st.markdown("---")
    st.markdown("## Performance do Sistema (conferência automática)")
    if st.session_state.accuracy_log:
        total = len(st.session_state.accuracy_log)
        acertos = sum(st.session_state.accuracy_log)
        taxa = (acertos / total) * 100
        st.markdown(f"- **Total de sinais avaliados:** {total}")
        st.markdown(f"- **Acertos:** {acertos}")
        st.markdown(f"- **Taxa de acerto:** <span style='font-size:24px; color: {'green' if taxa >= 60 else 'orange' if taxa >= 50 else 'red'};'>**{taxa:.2f}%**</span>", unsafe_allow_html=True)
        
        with st.expander("Últimas 20 Conferências (Acertos/Erros)"):
            if st.session_state.accuracy_log:
                ultimas = st.session_state.accuracy_log[-20:]
                # Ajusta o índice de exibição para começar do número correto da predição
                start_idx_display = len(st.session_state.accuracy_log) - len(ultimas) + 1 
                for i, acerto in enumerate(ultimas, start=start_idx_display):
                    st.write(f"#{i}: {'✅ Acertou' if acerto else '❌ Errou'}")
            else:
                st.write("Nenhuma conferência para exibir.")
    else:
        st.write("Ainda sem dados suficientes para avaliar desempenho do sistema.")
        st.info("Para começar a avaliar a performance, insira mais resultados.")

if __name__ == "__main__":
    main()
