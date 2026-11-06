"""
Mede a qualidade da BUSCA (sem chamar o LLM).

Para cada pergunta, confere se algum dos TOP_K trechos recuperados contém a
frase que responde a pergunta. Útil para comparar mudanças de chunking,
modelo de embeddings ou tipo de busca.

Uso: python avaliar_busca.py          (mostra só as falhas)
     python avaliar_busca.py -v       (mostra todas)
"""
import sys

from ask import TOP_K, retrieve

# (pergunta, trecho que precisa estar entre os recuperados)
# As perguntas usam palavras diferentes das dos documentos, de propósito.
CASOS = [
    ("O guia fala inglês?", "inglês ou espanhol para outros passeios"),
    ("Tem guia que fala espanhol?", "inglês ou espanhol para outros passeios"),
    ("Quanto custa o catamarã para criança de 8 anos?", "Criança de 6 a 11 anos: R$ 90"),
    ("Quanto é o mergulho?", "R$ 290 por pessoa"),
    ("Qual o valor da lancha para o fim de semana?", "acréscimo de 15%"),
    ("Tem desconto para grupo grande?", "Grupos a partir de 8 pessoas"),
    ("Grávida pode mergulhar?", "proibido para gestantes"),
    ("Eu enjoo em barco, o que faço?", "remédio para enjoo"),
    ("Esqueci meu óculos no ônibus", "Objetos esquecidos"),
    ("Qual horário do quiosque do shopping?", "Quiosque do Shopping"),
    ("Aceita dinheiro vivo?", "dinheiro só são aceitos nos quiosques"),
    ("Posso parcelar no cartão?", "até 6 vezes sem juros"),
    ("O Pix expirou, e agora?", "O link de Pix expira em 30 minutos"),
    ("Quanto tempo demora o estorno no cartão?", "até duas faturas"),
    ("Meu sobrinho menor de idade pode ir comigo?", "autorização simples"),
    ("A van espera se eu atrasar?", "espera no máximo 10 minutos"),
    ("Preciso levar documento?", "documento com foto"),
    ("Dá pra trocar o nome de quem vai no passeio?", "troca de nomes"),
    ("Quero fazer um aniversário no barco", "fretar o catamarã inteiro"),
    ("Tem passeio à noite?", "Tour gastronômico noturno"),
    ("Tem opção vegetariana no tour de comida?", "Pratos vegetarianos"),
    ("Quanto custa alugar sapatilha?", "sapatilha de neoprene: R$ 15"),
    ("Que roupa usar para entrar na igreja?", "ombros cobertos"),
    ("Precisa levar casaco no catamarã?", "casaco leve"),
    ("Quando é a melhor época para visitar?", "época mais seca"),
    ("Por que as piscinas naturais não saem todo dia?", "maré baixa"),
    ("Posso levar drone?", "drones é proibido"),
    ("Tem banheiro no catamarã?", "banheiro a bordo"),
    ("Cadeirante consegue fazer algum passeio?", "acessíveis para cadeirantes"),
    ("O seguro cobre celular quebrado?", "não cobre perda ou dano"),
    ("Se chover meu passeio de barco é cancelado?", "avaliação do comandante"),
    ("Desisti da viagem uma semana antes, recebo o dinheiro?", "reembolso integral"),
    ("Posso levar meu cachorro?", "Animais de estimação"),
    ("Quanto tempo dura o city tour?", "retorno por volta das 13h"),
    ("Onde almoça no passeio do litoral norte?", "restaurante parceiro"),
    ("Quantas pessoas cabem na lancha?", "até 10 pessoas"),
]


def main():
    verbose = "-v" in sys.argv
    acertos = 0
    for pergunta, esperado in CASOS:
        hits = retrieve(pergunta)
        pos = next((i for i, (_, c, _) in enumerate(hits) if esperado in c), None)
        ok = pos is not None
        acertos += ok
        if verbose or not ok:
            marca = f"OK (#{pos + 1})" if ok else "FALHOU"
            print(f"[{marca:8}] {pergunta}")
            if not ok:
                print(f"           esperado: {esperado!r}")
                print(f"           veio: {', '.join(s for s, _, _ in hits)}")
    total = len(CASOS)
    print(f"\nAcertos: {acertos}/{total} ({acertos / total:.0%}) com TOP_K={TOP_K}")


if __name__ == "__main__":
    main()
