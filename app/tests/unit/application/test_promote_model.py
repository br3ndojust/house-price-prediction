import pytest

from app.application.use_cases.promote_model import PromoteModel
from app.domain.entities.model_version import ModelVersion


class FakeModelRepository:
    def __init__(self):
        self.reloaded_with = None

    def reload(self, artifact_path):
        self.reloaded_with = artifact_path


class FakeModelVersionRepository:
    def __init__(self, versions: dict):
        self._versions = versions
        self.promoted = None

    def get(self, version):
        return self._versions.get(version)

    def promote(self, version, alias=None):
        self.promoted = (version, alias)
        v = self._versions[version]
        v.status = "active"
        return v


def test_promote_unknown_version_raises():
    use_case = PromoteModel(FakeModelVersionRepository({}), FakeModelRepository())
    with pytest.raises(ValueError, match="não encontrada"):
        use_case.execute("does-not-exist")


def test_promote_fails_clearly_when_artifact_file_missing(tmp_path):
    missing_path = tmp_path / "gone.pkl"
    version = ModelVersion(
        version="candidate-1", algorithm="xgboost", hyperparameters={}, artifact_path=str(missing_path),
        status="candidate",
    )
    models = FakeModelRepository()
    use_case = PromoteModel(FakeModelVersionRepository({"candidate-1": version}), models)

    with pytest.raises(ValueError, match="não encontrado"):
        use_case.execute("candidate-1")
    assert models.reloaded_with is None  # nunca chegou a tentar recarregar um arquivo inexistente


def test_promote_succeeds_when_artifact_exists(tmp_path):
    real_path = tmp_path / "model.pkl"
    real_path.write_bytes(b"fake-bytes")
    version = ModelVersion(
        version="candidate-1", algorithm="xgboost", hyperparameters={}, artifact_path=str(real_path),
        status="candidate",
    )
    versions_repo = FakeModelVersionRepository({"candidate-1": version})
    models = FakeModelRepository()
    use_case = PromoteModel(versions_repo, models)

    promoted = use_case.execute("candidate-1", alias="apelido")
    assert promoted.status == "active"
    assert versions_repo.promoted == ("candidate-1", "apelido")
    assert models.reloaded_with == str(real_path)
