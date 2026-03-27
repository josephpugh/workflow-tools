def test_end_to_end_disambiguation_then_input_collection(client) -> None:
    first = client.post(
        "/api/v1/conversations/turn",
        json={"message": "Update David's address"},
    )
    assert first.status_code == 200
    first_payload = first.json()
    assert first_payload["status"] == "needs_disambiguation"
    assert {
        first_payload["candidate_workflows"][0]["workflow"]["workflow_id"],
        first_payload["candidate_workflows"][1]["workflow"]["workflow_id"],
    } == {"update_client_mailing_address", "update_client_billing_address"}

    second = client.post(
        "/api/v1/conversations/turn",
        json={"session_id": first_payload["session_id"], "message": "Use the mailing address workflow"},
    )
    assert second.status_code == 200
    second_payload = second.json()
    assert second_payload["status"] == "needs_inputs"
    assert second_payload["selected_workflow"]["workflow_id"] == "update_client_mailing_address"
    assert second_payload["collected_inputs"]["client_name"] == "David"
    assert set(second_payload["missing_fields"]) == {"street_address", "city", "state", "postal_code", "effective_date"}
    assert "David" in second_payload["assistant_message"]
    assert "mailing address" in second_payload["assistant_message"].lower()
    assert "street" in second_payload["assistant_message"].lower()
    assert {field["name"] for field in second_payload["requested_fields"]} == {
        "street_address",
        "city",
        "state",
        "postal_code",
    }

    third = client.post(
        "/api/v1/conversations/turn",
        json={
            "session_id": first_payload["session_id"],
            "message": "Use 22 Broad St, Boston, MA 02110 effective April 1, 2026.",
        },
    )
    assert third.status_code == 200
    third_payload = third.json()
    assert third_payload["status"] == "ready"
    assert third_payload["selected_workflow"]["workflow_id"] == "update_client_mailing_address"
    assert third_payload["collected_inputs"] == {
        "client_name": "David",
        "street_address": "22 Broad St",
        "city": "Boston",
        "state": "MA",
        "postal_code": "02110",
        "effective_date": "2026-04-01",
    }
    assert third_payload["executable_contract"]["gathered_inputs"]["effective_date"] == "2026-04-01"


def test_disambiguation_preserves_original_details_in_first_needs_inputs_prompt(client) -> None:
    first = client.post(
        "/api/v1/conversations/turn",
        json={"message": "Update Dave Smith's address to 117 Hayworth Drive"},
    )
    assert first.status_code == 200
    first_payload = first.json()
    assert first_payload["status"] == "needs_disambiguation"

    second = client.post(
        "/api/v1/conversations/turn",
        json={"session_id": first_payload["session_id"], "message": "Use the mailing address workflow"},
    )
    assert second.status_code == 200
    second_payload = second.json()
    assert second_payload["status"] == "needs_inputs"
    assert second_payload["selected_workflow"]["workflow_id"] == "update_client_mailing_address"
    assert second_payload["collected_inputs"]["client_name"] == "Dave Smith"
    assert second_payload["collected_inputs"]["street_address"] == "117 Hayworth Drive"
    assert "Dave Smith" in second_payload["assistant_message"]
    assert "117 Hayworth Drive" in second_payload["assistant_message"]
    assert "city" in second_payload["assistant_message"].lower()
    assert "postal code" in second_payload["assistant_message"].lower()


def test_first_needs_inputs_prompt_mentions_partial_address_context_once(client) -> None:
    first = client.post(
        "/api/v1/conversations/turn",
        json={"message": "Update Dave Smith's address to Chapel Hill, NC 27517"},
    )
    assert first.status_code == 200
    first_payload = first.json()
    assert first_payload["status"] == "needs_disambiguation"

    second = client.post(
        "/api/v1/conversations/turn",
        json={
            "session_id": first_payload["session_id"],
            "message": "Thanks, yeah I want to update the client mailing address",
        },
    )
    assert second.status_code == 200
    second_payload = second.json()
    assert second_payload["status"] == "needs_inputs"
    assert second_payload["collected_inputs"] == {
        "client_name": "Dave Smith",
        "city": "Chapel Hill",
        "state": "NC",
        "postal_code": "27517",
    }
    assert "Dave Smith" in second_payload["assistant_message"]
    assert "Chapel Hill" in second_payload["assistant_message"]
    assert "27517" in second_payload["assistant_message"]
    assert "street address" in second_payload["assistant_message"].lower()

    third = client.post(
        "/api/v1/conversations/turn",
        json={
            "session_id": first_payload["session_id"],
            "message": "The street address is 117 Hayworth Drive and it should take effect on 2026-04-01",
        },
    )
    assert third.status_code == 200
    third_payload = third.json()
    assert third_payload["status"] == "ready"
    assert "I already have Chapel Hill, NC, 27517" not in third_payload["assistant_message"]


def test_end_to_end_one_shot_vendor_payment(client) -> None:
    response = client.post(
        "/api/v1/conversations/turn",
        json={
            "message": "Schedule a $4500 vendor payment to Acme Supplies on 2026-04-15 from operating account for invoice INV-1001",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["selected_workflow"]["workflow_id"] == "schedule_vendor_payment"
    assert payload["collected_inputs"] == {
        "payee_name": "Acme Supplies",
        "amount": 4500,
        "payment_date": "2026-04-15",
        "source_account": "operating account",
        "invoice_number": "INV-1001",
    }


def test_unsupported_request_returns_no_candidates(client) -> None:
    response = client.post(
        "/api/v1/conversations/turn",
        json={
            "message": "Book a trade for 500 dollars for .SPX for this guy",
            "context": {"client_name": "David Smith"},
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "unsupported"
    assert payload["selected_workflow"] is None
    assert payload["candidate_workflows"] == []
    assert "couldn’t identify a workflow" in payload["assistant_message"]


def test_end_to_end_context_prefill_and_batch_follow_up(client) -> None:
    first = client.post(
        "/api/v1/conversations/turn",
        json={
            "message": "Generate a monthly portfolio report",
            "context": {"client_name": "Alice Johnson"},
        },
    )
    assert first.status_code == 200
    first_payload = first.json()
    assert first_payload["status"] == "needs_inputs"
    assert first_payload["selected_workflow"]["workflow_id"] == "generate_monthly_portfolio_report"
    assert first_payload["collected_inputs"]["client_name"] == "Alice Johnson"
    assert set(first_payload["missing_fields"]) == {"report_month", "delivery_channel"}

    second = client.post(
        "/api/v1/conversations/turn",
        json={
            "session_id": first_payload["session_id"],
            "message": "Report month is 2026-02 and send it through email",
        },
    )
    assert second.status_code == 200
    second_payload = second.json()
    assert second_payload["status"] == "ready"
    assert second_payload["collected_inputs"] == {
        "client_name": "Alice Johnson",
        "report_month": "2026-02-01",
        "delivery_channel": "email",
    }


def test_relative_date_input_is_resolved_using_current_date(client) -> None:
    first = client.post(
        "/api/v1/conversations/turn",
        json={"message": "Update Dave Smith's client mailing address to Chapel Hill, NC, 27517"},
    )
    assert first.status_code == 200
    first_payload = first.json()
    assert first_payload["status"] == "needs_inputs"

    second = client.post(
        "/api/v1/conversations/turn",
        json={
            "session_id": first_payload["session_id"],
            "message": "The street address is 117 Hayworth Drive and it should take effect next Wednesday",
        },
    )
    assert second.status_code == 200
    second_payload = second.json()
    assert second_payload["status"] == "ready"
    assert second_payload["collected_inputs"]["effective_date"] == "2026-04-01"
    assert second_payload["executable_contract"]["gathered_inputs"]["effective_date"] == "2026-04-01"


def test_meeting_workflow_supports_start_time_and_duration(client) -> None:
    response = client.post(
        "/api/v1/conversations/turn",
        json={
            "message": "Book a client meeting with Alice Johnson next Wednesday at 11:30 for 45 minutes about quarterly planning",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "needs_inputs"
    assert payload["selected_workflow"]["workflow_id"] == "book_client_meeting"
    assert payload["collected_inputs"]["meeting_date"] == "2026-04-01"
    assert payload["collected_inputs"]["meeting_start_time"] == "11:30"
    assert payload["collected_inputs"]["duration_minutes"] == 45
    assert payload["collected_inputs"]["client_name"] == "Alice Johnson"
    assert payload["collected_inputs"]["agenda"] == "Quarterly planning"
    assert "meeting start time" not in " ".join(payload["missing_fields"])
    assert "duration_minutes" not in payload["missing_fields"]


def test_meeting_workflow_can_suggest_slots(client) -> None:
    response = client.post(
        "/api/v1/conversations/turn",
        json={
            "message": "Book a client meeting with Alice Johnson for 45 minutes virtual about quarterly planning",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "needs_choice"
    assert payload["selected_workflow"]["workflow_id"] == "book_client_meeting"
    assert payload["collected_inputs"]["client_name"] == "Alice Johnson"
    assert payload["collected_inputs"]["duration_minutes"] == 45
    assert payload["collected_inputs"]["meeting_format"] == "virtual"
    assert payload["collected_inputs"]["agenda"] == "Quarterly planning"
    assert len(payload["choices"]) == 3
    assert payload["choices"][0]["value"]["meeting_start_time"] == "10:00"


def test_meeting_workflow_suggests_times_for_known_date(client) -> None:
    response = client.post(
        "/api/v1/conversations/turn",
        json={
            "message": "Help me book a meeting with Dave Smith for next wednesday",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "needs_choice"
    assert payload["selected_workflow"]["workflow_id"] == "book_client_meeting"
    assert payload["collected_inputs"]["meeting_date"] == "2026-04-01"
    assert len(payload["choices"]) == 3
    assert {choice["value"]["meeting_date"] for choice in payload["choices"]} == {"2026-04-01"}
    assert payload["choices"][0]["value"]["meeting_start_time"] == "10:00"


def test_meeting_workflow_validates_booked_slots_and_offers_alternatives(client) -> None:
    first = client.post(
        "/api/v1/conversations/turn",
        json={
            "message": "Book a client meeting with Alice Johnson next Wednesday at 14:30 for 45 minutes virtual about quarterly planning",
        },
    )
    assert first.status_code == 200
    first_payload = first.json()
    assert first_payload["status"] == "needs_choice"
    assert first_payload["selected_workflow"]["workflow_id"] == "book_client_meeting"
    assert first_payload["validation_result"]["result"] == "failed"
    assert len(first_payload["choices"]) >= 1

    second = client.post(
        "/api/v1/conversations/turn",
        json={
            "session_id": first_payload["session_id"],
            "message": "The first option works for me",
        },
    )
    assert second.status_code == 200
    second_payload = second.json()
    assert second_payload["status"] == "ready"
    assert second_payload["selected_workflow"]["workflow_id"] == "book_client_meeting"
    assert second_payload["collected_inputs"]["meeting_start_time"] == "10:00"
    assert second_payload["collected_inputs"]["meeting_date"] == "2026-04-01"


def test_ready_session_allows_field_updates(client) -> None:
    first = client.post(
        "/api/v1/conversations/turn",
        json={"message": "Update David's address"},
    )
    first_payload = first.json()

    client.post(
        "/api/v1/conversations/turn",
        json={"session_id": first_payload["session_id"], "message": "Use the mailing address workflow"},
    )

    third = client.post(
        "/api/v1/conversations/turn",
        json={
            "session_id": first_payload["session_id"],
            "message": "Use 22 Broad St, Boston, MA 02110 effective April 1, 2026.",
        },
    )
    third_payload = third.json()
    assert third_payload["status"] == "ready"
    assert third_payload["collected_inputs"]["effective_date"] == "2026-04-01"

    fourth = client.post(
        "/api/v1/conversations/turn",
        json={
            "session_id": first_payload["session_id"],
            "message": "Change effective date to 2026-05-15",
        },
    )
    assert fourth.status_code == 200
    fourth_payload = fourth.json()
    assert fourth_payload["status"] == "ready"
    assert fourth_payload["collected_inputs"]["effective_date"] == "2026-05-15"
    assert "updated effective_date" in fourth_payload["assistant_message"].lower()


def test_direct_match_acknowledges_known_details_in_first_follow_up(client) -> None:
    response = client.post(
        "/api/v1/conversations/turn",
        json={"message": "Update Dave Smith's mailing address to 117 Hayworth Drive"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "needs_inputs"
    assert payload["selected_workflow"]["workflow_id"] == "update_client_mailing_address"
    assert payload["collected_inputs"]["client_name"] == "Dave Smith"
    assert payload["collected_inputs"]["street_address"] == "117 Hayworth Drive"
    assert "Dave Smith" in payload["assistant_message"]
    assert "117 Hayworth Drive" in payload["assistant_message"]
    assert "city" in payload["assistant_message"].lower()
