from investigation_rag.entities import ParsedEntity
from investigation_rag.entity_evaluation import evaluate_required_entities


def test_required_entity_evaluation_does_not_treat_unannotated_entities_as_false_positives():
    truth = {
        "entities": [
            {"text": "Gemeente Enschede", "type": "organization"},
            {
                "text": "Ministerie van Infrastructuur en Waterstaat",
                "type": "department",
                "accepted_types": ["department", "organization"],
            },
            {"text": "Enschede", "type": "location"},
        ]
    }
    extracted = [
        ParsedEntity("gemeente enschede", "organization"),
        ParsedEntity("Ministerie van Infrastructuur en Waterstaat", "organization"),
        ParsedEntity("Extra correct–entity", "person"),
    ]

    result = evaluate_required_entities(extracted, truth)

    assert result.required_total == 3
    assert result.names_found == 2
    assert result.correct_type == 2
    assert result.missed == ("Enschede",)
    assert result.wrong_type == ()
    assert result.unscored_extra == ("Extra correct–entity [person]",)
