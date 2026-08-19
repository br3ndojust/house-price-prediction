from __future__ import annotations

from pathlib import Path

from app.domain.entities.model_version import ModelVersion
from app.domain.ports.model_repository import ModelRepository
from app.domain.ports.model_version_repository import ModelVersionRepository


class PromoteModel:
    """Aprova e ativa um candidato — nunca automático (P6, docs/08_continuous_learning.md seção 6)."""

    def __init__(self, model_version_repository: ModelVersionRepository, model_repository: ModelRepository):
        self._versions = model_version_repository
        self._models = model_repository

    def execute(self, version: str, alias: str | None = None) -> ModelVersion:
        candidate = self._versions.get(version)
        if candidate is None:
            raise ValueError(f"versão {version} não encontrada")
        if not Path(candidate.artifact_path).exists():
            # a linha em `model_versions` sobrevive (SQLite persistido em volume), mas o .pkl é um
            # arquivo solto em artifacts/candidates/ — some se alguém limpar aquele diretório (ou o
            # bind mount for compartilhado com outro processo que rode `rm -rf`). Falha ANTES de
            # marcar a versão como "active" no banco — nunca deixa o estado inconsistente.
            raise ValueError(
                f"artefato de '{version}' não encontrado em {candidate.artifact_path} — o arquivo "
                "foi removido do disco (o registro no banco sobrevive, o .pkl não). Rode o treino de "
                "novo para gerar um candidato novo antes de promover."
            )
        promoted = self._versions.promote(version, alias)
        self._models.reload(promoted.artifact_path)
        return promoted
