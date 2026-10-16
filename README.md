# agent-bench

Harnais de benchmark par **ablation** pour un agent qui interroge une base de données avec des outils. Tu définis une variante de référence, tu changes un élément à la fois (version du glossaire, données enrichies, outils disponibles, prompt, modèle…), et le harnais mesure ce que chaque changement apporte, avec des intervalles de confiance.

```
experiment.yaml ──► 1. RUN ──► traces/ (une par run, en cache) ──► 2. GRADE ──► scores/ (en cache) ──► 3. REPORT
   variantes          agent + outils                                  scoreurs                      report.md + scores.csv
```

Les trois étapes sont indépendantes :

- Changer la grille du juge relance **seulement la notation**, jamais l'agent.
- Une run est identifiée par un hash de la variante (paramètres de l'agent, texte du prompt, **contenu** des artefacts, descriptions des outils), de la question et du numéro d'essai. Si tu modifies un fichier de glossaire sans changer son numéro de version, les runs qui l'utilisent sont relancées, et uniquement celles-là.

## Démarrage

```bash
uv venv && uv pip install -e ".[dev]"
pytest                                                 # tests, sans réseau

bench all examples/demo/experiment.offline.yaml        # pipeline complet avec des agents scriptés, sans clé API

export ANTHROPIC_API_KEY=...                           # ou `ant auth login`
bench plan examples/demo/experiment.yaml               # ce qui va tourner, ce qui est déjà en cache
bench all  examples/demo/experiment.yaml               # run + notation + rapport
bench all  examples/demo/experiment.yaml --variants baseline,glossary_v1 --split dev
```

Le rapport est écrit dans `examples/demo/runs/<experiment>/report.md`. Il contient :

- un classement des variantes : métrique principale avec IC 95 %, pass^k, écart avec la baseline apparié par question, p-value, questions gagnées / perdues / à égalité, coût, nombre d'étapes ;
- le taux de réussite par critère du juge ;
- les résultats par catégorie de questions et par question ;
- la **liste des échecs à lire**, avec un lien vers chaque trace.

`scores.csv` contient tous les scores au format long (une ligne par score), prêt pour pandas.

## Organisation du code

```
src/agent_bench/
  config.py        Variant, Ablation, grille factorielle, chargement YAML
  artifacts.py     artefacts versionnés "nom@version", hash de contenu
  dataset.py       Sample : question, références, catégorie, split dev/holdout
  db.py            base figée en lecture seule (SQLite ; autres moteurs via le registre DATABASES)
  tools/           registre TOOLS : chaque outil est construit à partir des artefacts de la variante
  agents/          registre AGENTS : claude (boucle d'appels d'outils), scripted (tests)
  trace.py         ce qu'une run laisse derrière elle : appels d'outils, tokens, fin de run, erreurs
  runner.py        étape 1 : plan + exécution en parallèle + cache
  scorers/         registre SCORERS : result_match, answer_values, tables, efficiency, rubric_judge
  grading.py       étape 2 : notation + cache par empreinte de scoreur
  analysis.py      bootstrap par question, comparaison appariée, pass^k, test de permutation
  report.py        étape 3 : rapport markdown + CSV
  calibration.py   accord juge / annotations humaines (kappa de Cohen)
examples/demo/     base e-commerce, glossaires v1/v2, tables enrichies, flux, 7 questions
```

Chaque extension se fait en **ajoutant une fonction ou une classe enregistrée**, sans toucher au reste.

## Brancher tes données

1. **La base.** Pointe `database.path` vers un *snapshot figé*. Pour un autre moteur (Postgres, BigQuery…), implémente le protocole `Database` (`execute`, `table_names`) et enregistre-le avec `@DATABASES.register("postgres")`.
2. **Les artefacts.** Dépose tes sorties de pipeline dans `artifacts/<nom>/<version>.yaml|json|md`, par exemple `artifacts/glossary/v3.yaml` ou `artifacts/tables/enrich_gpt_v2.json`. Une variante les référence par `glossary: glossary@v3`.
3. **Les outils.** Un outil est une fabrique qui reçoit les artefacts et la base :

   ```python
   from agent_bench.tools import TOOLS, Tool, ToolContext, ToolError, object_schema

   @TOOLS.register("lookup_operation")
   def lookup_operation(ctx: ToolContext) -> Tool:
       ops = ctx.artifacts["operations"].data
       def fn(name: str) -> dict:
           if name not in ops:
               raise ToolError(f"Unknown operation '{name}'")   # renvoyée à l'agent comme erreur récupérable
           return ops[name]
       return Tool(name="lookup_operation", description="...",
                   input_schema=object_schema({"name": {"type": "string"}}),
                   fn=fn, requires=["operations"])
   ```

   La description de l'outil fait partie de l'empreinte de la variante : si tu la reformules, ça compte comme une nouvelle variante. C'est voulu, car la description change le comportement de l'agent.
4. **Les questions.** Une ligne JSON par question dans `dataset.jsonl` : `question`, `reference_sql` (comparée par exécution), `reference_answer` (pour le juge), `expected_values`, `expected_tables`, `category`, `answerable`, `split`.
5. **Un autre agent** (un autre fournisseur, LangGraph, une architecture planificateur / exécutant…) : une classe avec `run(question, system_prompt, tools) -> AgentResult`, enregistrée avec `@AGENTS.register("mon_agent")`, puis `agent: {kind: mon_agent, ...}` dans la config.

## Les métriques

| Scoreur | Ce qu'il mesure | LLM ? |
|---|---|---|
| `result_match` | Exécute le SQL de référence et la dernière requête de l'agent, puis compare les résultats (ordre des lignes et noms de colonnes ignorés, colonnes en plus tolérées). | non |
| `answer_values` | Part des `expected_values` présentes dans la réponse finale (formats de nombres FR et EN acceptés). | non |
| `tables` | Rappel et précision des tables consultées par rapport à `expected_tables`. | non |
| `efficiency` | Appels d'outils, erreurs d'outils, tours, tokens, coût, latence. | non |
| `rubric_judge` | Critères binaires d'une grille (`correct`, `grounded`, `business_rules`, `abstains`, `concise`) ; le raisonnement est écrit avant chaque verdict. | oui |

Les métriques déterministes sont la vérité de référence. Le juge sert à ce qu'on ne peut pas vérifier par exécution : questions sans réponse, explications, règles métier citées.

## La boucle d'amélioration

1. `bench all … --split dev` sur la baseline.
2. Lire les traces listées dans **Failures to read** et noter la cause de chaque échec (mauvaise table, jointure ratée, terme mal compris, hallucination…).
3. Corriger ce qui revient le plus souvent : glossaire, enrichissement, description d'un outil, prompt.
4. Ajouter la correction comme **une nouvelle ablation**, relancer (seule la nouvelle variante tourne), puis comparer avec la baseline.
5. Ne regarder le split `holdout` qu'à la fin, pour vérifier que le gain se généralise.

## Calibrer le juge avant de lui faire confiance

```bash
bench export-labels examples/demo/experiment.yaml --n 40   # écrit runs/<exp>/labels.csv
# remplis les colonnes human.* (1/0) sans regarder les colonnes judge.*
bench calibrate examples/demo/runs/demo/labels.csv          # accord et kappa par critère, plus les désaccords
```

Un kappa de 0,6 ou plus est utilisable, 0,8 ou plus est bon. En dessous, reformule le critère en t'appuyant sur les désaccords.

## Points à connaître

- **Pas de « température 0 ».** Les modèles Claude actuels refusent les paramètres d'échantillonnage. Ici, la reproductibilité du juge vient du cache (chaque verdict est stocké sous l'empreinte modèle + grille + prompt) et, si besoin, d'un vote majoritaire (`votes: 3`).
- **Repli automatique en cas de refus.** Il est activé pour l'agent et pour le juge (`fallbacks: "default"`) : si le modèle refuse une requête pour raison de sécurité, l'API la relance sur un autre modèle. Ça fausserait une comparaison, donc chaque trace enregistre `fallback_used` et le rapport affiche le nombre de runs concernées. Mets `fallbacks: false` dans `agent:` pour le désactiver.
- **Taille d'échantillon.** Avec 25 questions, un écart de quelques points est souvent du bruit. Regarde l'IC de Δ et la p-value plutôt que les moyennes seules, et vise 50 à 100 questions.
- **Les runs en erreur** (quota d'API dépassé, panne…) sont exclues des scores, comptées dans le rapport, et relancées au prochain `bench run`.
