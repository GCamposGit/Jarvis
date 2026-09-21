# Jarvis Mission

## Objective

Evoluir o Jarvis de um assistente conversacional reativo para um Sistema Operacional Autônomo de Produtividade Executiva e Operações de Negócios (Autonomous Business & Executive Operating System), equipado com um Agent Harness determinístico, gerenciamento de Segundo Cérebro de longo prazo via MCP, orquestração de subagentes especializados e roteamento multi-modelo de custo ótimo (Local-First Ollama + Fronteira de Pareto em Nuvem).

## Non-goals

1. Não substituir ferramentas corporativas de mercado (ERP, CRM, Slack, Google Workspace), mas sim orquestrá-las e integrá-las de forma headless via MCP e APIs seguras.
2. Não executar transações financeiras irreversíveis ou modificações destrutivas sem aprovação explícita do operador humano (enforçar Autonomy Level 2 / HITL).
3. Não armazenar credenciais, segredos, chaves de API ou dados confidenciais diretamente no histórico conversacional ou em artefatos rastreados pelo Git.
4. Não depender exclusivamente de modelos de nuvem proprietários de alto custo para tarefas rotineiras de triagem, formatação e verificação.

## Success criterion

1. Arquitetura 100% desacoplada e testável de forma headless (via biblioteca Python e APIs REST/WebSocket FastAPI).
2. Validação determinística contínua aprovada com zero falhas no harness oficial da Dark Factory (`python .factory/darkfac.py harness --quick`).
3. Suporte a memória de longo prazo com persistência estruturada (episódica e semântica) e rollback via MCP.
4. Roteamento dinâmico na Fronteira de Pareto que garanta custo operacional de inferência mínimo, priorizando o cluster local Ollama ($0) e modelos de alta eficiência ($0.005/task).
