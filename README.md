# Duvidas
## Eventos
1) Por que apenas "transaction" possui valor?
2) Por que apenas "transaction" não possui "offer_id"?

Soluçao: pedi ajuda a IA para me explicar o motivo para esses detalhes

# Detalhes
1) Daqueles que não completaram uma transação, apenas 20 não receberam oferta (time_since_test_start_recv = NaN)

2) time_since alguma coisa:
time_since_test_start_transaction   = 0.0   → transação aconteceu no início do experimento
time_since_test_start_recv          = 0.0   → oferta foi recebida no início do experimento
time_since_test_start_view          = 0.0   → oferta foi vista no início do experimento
time_since_test_start_jornada       = NaN   → oferta NÃO foi completada

Quando a jornada é NaN, então a oferta não foi utilizada. Como houve uma transação, ela não foi o suficiente para utilizar a oferta, talvez o valor da transação não tenha sido suficiente.

3) Datas
Eu poderia utilizar as datas de registro para calcular a idade do usuário com relação a oferta, por exemplo, mas não sei a data da oferta.

4) Propensão a utilização de oferta
Pensei que utilizar a porcentagem de sucesso de uma oferta por usuário. Conversando com o "Claúdio", ele apontou que isso poderia trazer vazamento de dados, pois eu já indicaria ao modelo que o usuário tem maior chance de aceitar a oferta.
Se eu fosse prever o impacto de uma oferta futura, eu poderia utilizar o histórico para estimar esse valor.

5) Idade do usuário na criação da conta
Idade tem máximo de 118 anos. Ou seja, não é confiável e vai ser filtrada até 80

6) Channels
Se não utilizar a web, o sucesso é menor que o fracasso.

7) Fazer:
A partir dos clusters, ver qual oferta é melhor para cada grupo
Ver qual oferta converteu mais (caracteristicas, clustering de novo?)
