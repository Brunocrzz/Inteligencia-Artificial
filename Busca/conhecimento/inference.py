def forward_chaining(facts, rules):

    conclusions = []
    activated_rules = []

    for rule in rules:

        valid = True

        # Verifica condições positivas
        for condition in rule.get("if", []):
            if condition not in facts:
                valid = False
                break

        # Verifica condições negativas
        for condition in rule.get("not", []):
            if condition in facts:
                valid = False
                break

        if valid:
            conclusions.append(rule["then"])
            activated_rules.append(rule)

    # Se não houver nenhuma inferência define o diagnostico como inconclusivo
    if len(conclusions) == 0:
        conclusions.append("diagnostico_inconclusivo")
        activated_rules.append({
            "then": "diagnostico_inconclusivo",
            "confidence": 0
        })

    return conclusions, activated_rules