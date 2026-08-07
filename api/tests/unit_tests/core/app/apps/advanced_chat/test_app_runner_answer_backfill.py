"""Tests for the runner-thread answer backfill in AdvancedChatAppRunner."""

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

from sqlalchemy.dialects import sqlite

from core.app.apps.advanced_chat.app_runner import AdvancedChatAppRunner
from core.workflow.graph_events import GraphRunPartialSucceededEvent, GraphRunPausedEvent, GraphRunSucceededEvent
from core.workflow.runtime import GraphRuntimeState


def _build_runner(
    message_id: str = "message-id",
    sensitive_word_avoidance: object | None = None,
    workflow_run_id: str = "workflow-run-id",
) -> AdvancedChatAppRunner:
    runner = object.__new__(AdvancedChatAppRunner)
    runner.message = MagicMock()
    runner.message.id = message_id
    app_config = MagicMock()
    app_config.tenant_id = "tenant-id"
    app_config.app_id = "app-id"
    app_config.sensitive_word_avoidance = sensitive_word_avoidance
    generate_entity = MagicMock()
    generate_entity.app_config = app_config
    generate_entity.workflow_run_id = workflow_run_id
    runner.application_generate_entity = generate_entity
    runner._queue_manager = MagicMock()
    return runner


def _patch_db_engine():
    mock_db = MagicMock()
    mock_db.engine = MagicMock()
    return patch("core.app.apps.advanced_chat.app_runner.db", mock_db)


def _compile_update_sql(stmt) -> str:
    return str(stmt.compile(dialect=sqlite.dialect(), compile_kwargs={"literal_binds": True}))


class TestBackfillMessageAnswer:
    """Test that the runner thread backfills an empty message answer from workflow outputs."""

    def test_backfills_empty_answer_from_workflow_outputs(self) -> None:
        # Arrange
        runner = _build_runner()
        runtime_state = MagicMock(spec=GraphRuntimeState)
        runtime_state.outputs = {"answer": "answer from workflow outputs"}
        usage = MagicMock()
        usage.prompt_tokens = 100
        usage.prompt_unit_price = 0.001
        usage.prompt_price_unit = 1000
        usage.completion_tokens = 50
        usage.completion_unit_price = 0.002
        usage.completion_price_unit = 1000
        usage.total_price = 0.2
        usage.currency = "USD"
        runtime_state.llm_usage = usage
        session = MagicMock()
        session.__enter__.return_value = session
        result = MagicMock()
        result.rowcount = 1
        session.execute.return_value = result

        # Act
        with (
            _patch_db_engine(),
            patch("core.app.apps.advanced_chat.app_runner.Session", return_value=session),
            patch("core.app.apps.advanced_chat.app_runner.logger") as mock_logger,
            patch("core.app.apps.advanced_chat.app_runner.time.perf_counter", side_effect=[10.5]),
        ):
            runner._backfill_message_answer(runtime_state, run_started_at=10.0)

        # Assert
        session.execute.assert_called_once()
        update_stmt = session.execute.call_args.args[0]
        compiled_sql = _compile_update_sql(update_stmt)
        assert "messages.answer = ''" in compiled_sql
        assert "messages.id = 'message-id'" in compiled_sql
        assert "answer='answer from workflow outputs'" in compiled_sql
        assert "workflow_run_id='workflow-run-id'" in compiled_sql
        assert "answer_tokens=50" in compiled_sql
        assert "message_tokens=100" in compiled_sql
        assert "answer_unit_price=0.002" in compiled_sql
        assert "answer_price_unit=1000" in compiled_sql
        assert "message_unit_price=0.001" in compiled_sql
        assert "message_price_unit=1000" in compiled_sql
        assert "total_price=0.2" in compiled_sql
        assert "currency='USD'" in compiled_sql
        assert "provider_response_latency=0.5" in compiled_sql
        session.commit.assert_called_once()
        mock_logger.warning.assert_called_once_with(
            "Backfilled an empty message answer from workflow outputs, message_id=%s",
            "message-id",
        )

    def test_backfills_answer_only_when_llm_usage_is_absent(self) -> None:
        # Arrange
        runner = _build_runner()
        runtime_state = MagicMock(spec=GraphRuntimeState)
        runtime_state.outputs = {"answer": "answer text"}
        runtime_state.llm_usage = None
        session = MagicMock()
        session.__enter__.return_value = session
        session.execute.return_value.rowcount = 1

        # Act
        with (
            _patch_db_engine(),
            patch("core.app.apps.advanced_chat.app_runner.Session", return_value=session),
            patch("core.app.apps.advanced_chat.app_runner.logger"),
        ):
            runner._backfill_message_answer(runtime_state)

        # Assert
        update_stmt = session.execute.call_args.args[0]
        compiled_sql = _compile_update_sql(update_stmt)
        assert "answer='answer text'" in compiled_sql
        assert "answer_tokens" not in compiled_sql
        assert "provider_response_latency" not in compiled_sql

    def test_skips_warning_when_answer_was_already_persisted(self) -> None:
        # Arrange
        runner = _build_runner()
        runtime_state = MagicMock(spec=GraphRuntimeState)
        runtime_state.outputs = {"answer": "answer from workflow outputs"}
        session = MagicMock()
        session.__enter__.return_value = session
        result = MagicMock()
        result.rowcount = 0
        session.execute.return_value = result

        # Act
        with (
            _patch_db_engine(),
            patch("core.app.apps.advanced_chat.app_runner.Session", return_value=session),
            patch("core.app.apps.advanced_chat.app_runner.logger") as mock_logger,
        ):
            runner._backfill_message_answer(runtime_state)

        # Assert
        session.execute.assert_called_once()
        session.commit.assert_called_once()
        mock_logger.warning.assert_not_called()

    @patch("core.app.apps.advanced_chat.app_runner.db", MagicMock())
    @patch("core.app.apps.advanced_chat.app_runner.Session")
    def test_skips_update_when_outputs_have_no_answer(self, mock_session: MagicMock) -> None:
        # Arrange
        runner = _build_runner()
        runtime_state = MagicMock(spec=GraphRuntimeState)
        runtime_state.outputs = {}

        # Act
        runner._backfill_message_answer(runtime_state)

        # Assert
        mock_session.return_value.__enter__.return_value.execute.assert_not_called()

    @patch("core.app.apps.advanced_chat.app_runner.db", MagicMock())
    @patch("core.app.apps.advanced_chat.app_runner.Session")
    def test_skips_update_when_answer_is_blank(self, mock_session: MagicMock) -> None:
        # Arrange
        runner = _build_runner()
        runtime_state = MagicMock(spec=GraphRuntimeState)
        runtime_state.outputs = {"answer": "   "}

        # Act
        runner._backfill_message_answer(runtime_state)

        # Assert
        mock_session.return_value.__enter__.return_value.execute.assert_not_called()

    @patch("core.app.apps.advanced_chat.app_runner.db", MagicMock())
    @patch("core.app.apps.advanced_chat.app_runner.Session")
    def test_skips_update_when_answer_is_not_a_string(self, mock_session: MagicMock) -> None:
        # Arrange
        runner = _build_runner()
        runtime_state = MagicMock(spec=GraphRuntimeState)
        runtime_state.outputs = {"answer": {"text": "structured output"}}

        # Act
        runner._backfill_message_answer(runtime_state)

        # Assert
        mock_session.return_value.__enter__.return_value.execute.assert_not_called()

    def test_update_targets_only_the_message_row_with_empty_answer(self) -> None:
        # Arrange
        runner = _build_runner(message_id="message-42")
        runtime_state = MagicMock(spec=GraphRuntimeState)
        runtime_state.outputs = {"answer": "answer text"}
        session = MagicMock()
        session.__enter__.return_value = session
        session.execute.return_value.rowcount = 0

        # Act
        with (
            _patch_db_engine(),
            patch("core.app.apps.advanced_chat.app_runner.Session", return_value=session),
        ):
            runner._backfill_message_answer(runtime_state)

        # Assert
        update_stmt = session.execute.call_args.args[0]
        compiled_sql = _compile_update_sql(update_stmt)
        assert "UPDATE messages SET" in compiled_sql
        assert "messages.id = 'message-42'" in compiled_sql
        assert "messages.answer = ''" in compiled_sql


class TestRunBackfillOnWorkflowCompletion:
    """Test that run() triggers the backfill after the workflow finishes."""

    def test_run_backfills_after_workflow_finishes(self) -> None:
        # Arrange
        runner = object.__new__(AdvancedChatAppRunner)
        runner.application_generate_entity = MagicMock()
        runner.application_generate_entity.app_config = MagicMock()
        runner.application_generate_entity.task_id = "task-id"
        runner.application_generate_entity.user_id = "user-id"
        runner.application_generate_entity.invoke_from = MagicMock()
        runner.application_generate_entity.query = "query"
        runner.application_generate_entity.inputs = {}
        runner.application_generate_entity.files = []
        runner.application_generate_entity.workflow_run_id = "workflow-run-id"
        runner.application_generate_entity.trace_manager = None
        runner.application_generate_entity.single_iteration_run = False
        runner.application_generate_entity.single_loop_run = False
        runner.conversation = MagicMock()
        runner.message = MagicMock()
        runner.message.id = "message-id"
        runner._workflow = MagicMock()
        runner._workflow.graph_dict = {}
        runner._workflow.id = "workflow-id"
        runner._workflow.tenant_id = "tenant-id"
        runner._workflow.app_id = "app-id"
        runner._workflow.type = "chat"
        runner._workflow.version = "1"
        runner._workflow.environment_variables = []
        runner._queue_manager = MagicMock()
        runner._graph_engine_layers = ()
        runner.system_user_id = "system-user-id"
        runner._app = MagicMock()
        runner._dialogue_count = 1
        runner._workflow_execution_repository = MagicMock()
        runner._workflow_node_execution_repository = MagicMock()

        runner.handle_input_moderation = MagicMock(return_value=False)
        runner.handle_annotation_reply = MagicMock(return_value=False)
        runner._initialize_conversation_variables = MagicMock(return_value=[])
        runner._init_graph = MagicMock(return_value=MagicMock())
        runner._handle_event = MagicMock()
        runner._backfill_message_answer = MagicMock()

        def fake_generator() -> Iterator[object]:
            yield GraphRunSucceededEvent()

        workflow_entry = MagicMock()
        workflow_entry.run.return_value = fake_generator()

        with (
            patch("core.app.apps.advanced_chat.app_runner.Session") as mock_session,
            patch("core.app.apps.advanced_chat.app_runner.SystemVariable"),
            patch("core.app.apps.advanced_chat.app_runner.VariablePool"),
            patch("core.app.apps.advanced_chat.app_runner.GraphRuntimeState") as mock_runtime_state_cls,
            patch("core.app.apps.advanced_chat.app_runner.RedisChannel"),
            patch("core.app.apps.advanced_chat.app_runner.WorkflowEntry", return_value=workflow_entry),
            patch("core.app.apps.advanced_chat.app_runner.WorkflowPersistenceLayer"),
            patch("core.app.apps.advanced_chat.app_runner.db") as mock_db,
        ):
            app_record = MagicMock()
            mock_session.return_value.__enter__.return_value.scalar.return_value = app_record
            mock_runtime_state = mock_runtime_state_cls.return_value

            # Act
            runner.run()

        # Assert
        runner._handle_event.assert_called_once()
        runner._backfill_message_answer.assert_called_once()
        assert runner._backfill_message_answer.call_args.args[0] is mock_runtime_state
        assert isinstance(runner._backfill_message_answer.call_args.args[1], float)

    def _run_with_events(self, *events: object) -> MagicMock:
        runner = object.__new__(AdvancedChatAppRunner)
        runner.application_generate_entity = MagicMock()
        runner.application_generate_entity.app_config = MagicMock()
        runner.application_generate_entity.task_id = "task-id"
        runner.application_generate_entity.user_id = "user-id"
        runner.application_generate_entity.invoke_from = MagicMock()
        runner.application_generate_entity.query = "query"
        runner.application_generate_entity.inputs = {}
        runner.application_generate_entity.files = []
        runner.application_generate_entity.workflow_run_id = "workflow-run-id"
        runner.application_generate_entity.trace_manager = None
        runner.application_generate_entity.single_iteration_run = False
        runner.application_generate_entity.single_loop_run = False
        runner.conversation = MagicMock()
        runner.message = MagicMock()
        runner.message.id = "message-id"
        runner._workflow = MagicMock()
        runner._workflow.graph_dict = {}
        runner._workflow.id = "workflow-id"
        runner._workflow.tenant_id = "tenant-id"
        runner._workflow.app_id = "app-id"
        runner._workflow.type = "chat"
        runner._workflow.version = "1"
        runner._workflow.environment_variables = []
        runner._queue_manager = MagicMock()
        runner._graph_engine_layers = ()
        runner.system_user_id = "system-user-id"
        runner._app = MagicMock()
        runner._dialogue_count = 1
        runner._workflow_execution_repository = MagicMock()
        runner._workflow_node_execution_repository = MagicMock()

        runner.handle_input_moderation = MagicMock(return_value=False)
        runner.handle_annotation_reply = MagicMock(return_value=False)
        runner._initialize_conversation_variables = MagicMock(return_value=[])
        runner._init_graph = MagicMock(return_value=MagicMock())
        runner._handle_event = MagicMock()
        runner._backfill_message_answer = MagicMock()

        def fake_generator() -> Iterator[object]:
            yield from events

        workflow_entry = MagicMock()
        workflow_entry.run.return_value = fake_generator()

        with (
            patch("core.app.apps.advanced_chat.app_runner.Session") as mock_session,
            patch("core.app.apps.advanced_chat.app_runner.SystemVariable"),
            patch("core.app.apps.advanced_chat.app_runner.VariablePool"),
            patch("core.app.apps.advanced_chat.app_runner.GraphRuntimeState"),
            patch("core.app.apps.advanced_chat.app_runner.RedisChannel"),
            patch("core.app.apps.advanced_chat.app_runner.WorkflowEntry", return_value=workflow_entry),
            patch("core.app.apps.advanced_chat.app_runner.WorkflowPersistenceLayer"),
            patch("core.app.apps.advanced_chat.app_runner.db"),
        ):
            mock_session.return_value.__enter__.return_value.scalar.return_value = MagicMock()
            runner.run()

        return runner

    def test_run_skips_backfill_without_success_terminal_event(self) -> None:
        # Act
        runner = self._run_with_events(MagicMock())

        # Assert
        runner._backfill_message_answer.assert_not_called()

    def test_run_skips_backfill_when_workflow_is_paused(self) -> None:
        # Act
        runner = self._run_with_events(GraphRunPausedEvent())

        # Assert
        runner._backfill_message_answer.assert_not_called()

    def test_run_backfills_after_partial_success(self) -> None:
        # Act
        runner = self._run_with_events(GraphRunPartialSucceededEvent(exceptions_count=1))

        # Assert
        runner._backfill_message_answer.assert_called_once()

    def test_run_backfills_after_success_followed_by_other_events(self) -> None:
        # Act
        runner = self._run_with_events(GraphRunSucceededEvent(), MagicMock())

        # Assert
        runner._backfill_message_answer.assert_called_once()


class TestBackfillOutputModeration:
    """Test that the backfilled answer goes through output moderation."""

    def test_backfill_applies_output_moderation_when_configured(self) -> None:
        # Arrange
        sensitive_word_avoidance = MagicMock()
        sensitive_word_avoidance.type = "direct_output"
        sensitive_word_avoidance.config = {"keywords": "blocked", "replacement": "***"}
        runner = _build_runner(sensitive_word_avoidance=sensitive_word_avoidance)
        runtime_state = MagicMock(spec=GraphRuntimeState)
        runtime_state.outputs = {"answer": "raw sensitive answer"}
        runtime_state.llm_usage = None
        session = MagicMock()
        session.__enter__.return_value = session
        session.execute.return_value.rowcount = 1

        # Act
        with (
            _patch_db_engine(),
            patch("core.app.apps.advanced_chat.app_runner.Session", return_value=session),
            patch("core.app.apps.advanced_chat.app_runner.OutputModeration") as mock_moderation_cls,
        ):
            mock_moderation = mock_moderation_cls.return_value
            mock_moderation.moderation_completion.return_value = ("moderated answer", True)
            runner._backfill_message_answer(runtime_state)

        # Assert
        mock_moderation_cls.assert_called_once()
        moderation_kwargs = mock_moderation_cls.call_args.kwargs
        assert moderation_kwargs["tenant_id"] == "tenant-id"
        assert moderation_kwargs["app_id"] == "app-id"
        assert moderation_kwargs["rule"].type == "direct_output"
        assert moderation_kwargs["rule"].config == {"keywords": "blocked", "replacement": "***"}
        mock_moderation.moderation_completion.assert_called_once_with(
            completion="raw sensitive answer", public_event=False
        )
        update_stmt = session.execute.call_args.args[0]
        compiled_sql = _compile_update_sql(update_stmt)
        assert "answer='moderated answer'" in compiled_sql
        assert "raw sensitive answer" not in compiled_sql

    def test_backfill_skips_output_moderation_when_not_configured(self) -> None:
        # Arrange
        runner = _build_runner()
        runtime_state = MagicMock(spec=GraphRuntimeState)
        runtime_state.outputs = {"answer": "plain answer"}
        runtime_state.llm_usage = None
        session = MagicMock()
        session.__enter__.return_value = session
        session.execute.return_value.rowcount = 1

        # Act
        with (
            _patch_db_engine(),
            patch("core.app.apps.advanced_chat.app_runner.Session", return_value=session),
            patch("core.app.apps.advanced_chat.app_runner.OutputModeration") as mock_moderation_cls,
        ):
            runner._backfill_message_answer(runtime_state)

        # Assert
        mock_moderation_cls.assert_not_called()
        update_stmt = session.execute.call_args.args[0]
        compiled_sql = _compile_update_sql(update_stmt)
        assert "answer='plain answer'" in compiled_sql
