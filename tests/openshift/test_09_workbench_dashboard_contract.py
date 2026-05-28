"""Level 1 — RHOAI dashboard contract for the PRAGMA Encoder workbench.

Verifies that the GitOps-managed workbench primitives are correctly shaped for
the RHOAI 3.4 dashboard.  All tests are read-only.

Two classes of defect are caught here (both discovered in production):

1. **ImageStream spec.tags missing** — the dashboard resolves selectable images
   from ``spec.tags``, not ``status.tags``.  An ImageStream with no ``spec.tags``
   shows as "image deleted" in the workbench image selector, even when the
   underlying image exists.  Every selectable tag must be declared in ``spec.tags``
   with a ``from.kind``/``from.name`` source reference.

2. **HardwareProfile invisible to dashboard** — the dashboard only surfaces
   profiles that exist in ``redhat-ods-applications`` and are not disabled.
   The ``opendatahub.io/dashboard-feature-visibility: '[]'`` annotation with an
   empty array must be absent; when present it was observed to suppress the
   profile from the selector.

Resources verified:
  - ImageStream ``pragma-encoder-workbench`` in ``redhat-ods-applications``
      - label ``opendatahub.io/notebook-image: "true"``
      - ``spec.tags`` non-empty
      - ``spec.tags[latest]`` exists with a ``from`` source reference
      - ``spec.tags[latest].annotations`` has ``opendatahub.io/notebook-software``
      - ``spec.tags[latest].annotations`` has ``opendatahub.io/workbench-image-recommended: "true"``
  - HardwareProfile ``pragma-encoder-gpu`` in ``redhat-ods-applications``
      - exists
      - ``opendatahub.io/disabled != "true"``
      - ``opendatahub.io/dashboard-feature-visibility`` annotation is absent
      - ``spec.identifiers`` contains an Accelerator entry for ``nvidia.com/gpu``
      - ``spec.scheduling.type == "Node"``
  - Notebook CR ``pragma-encoder-workbench`` in PRAGMA_TEST_NAMESPACE
      - annotation ``opendatahub.io/hardware-profile-name: pragma-encoder-gpu``
      - annotation ``opendatahub.io/hardware-profile-namespace: redhat-ods-applications``
      - annotation ``notebooks.opendatahub.io/last-image-selection`` references the IS tag
      - labels ``opendatahub.io/dashboard: "true"`` and ``opendatahub.io/odh-managed: "true"``

Prerequisites:
  - RUN_OPENSHIFT_TESTS=1
  - PRAGMA_TEST_NAMESPACE=pragma-encoder   (or whichever namespace the workbench lives in)
  - Level 0 tests passing (oc access)
"""

from __future__ import annotations

import os

import pytest

from tests.openshift.oc import oc_json, resource_exists

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RHOAI_NAMESPACE = "redhat-ods-applications"
_IS_NAME = "pragma-encoder-workbench"
_HP_NAME = "pragma-encoder-gpu"
_NB_NAME = "pragma-encoder-workbench"
_EXPECTED_TAG = "latest"


# ---------------------------------------------------------------------------
# TestImageStreamDashboardContract
# ---------------------------------------------------------------------------


class TestImageStreamDashboardContract:
    """ImageStream must be correctly shaped for RHOAI dashboard image discovery.

    The RHOAI dashboard resolves selectable notebook images from ImageStreams
    in ``redhat-ods-applications`` with label ``opendatahub.io/notebook-image=true``.
    It reads ``spec.tags`` — NOT ``status.tags`` — to populate the image selector.

    An ImageStream with ``spec.tags: []`` shows as 'image deleted' in the
    workbench image selector even when the underlying build succeeded and
    ``status.tags`` is populated.
    """

    def test_imagestream_exists_in_rhoai_namespace(self) -> None:
        """ImageStream pragma-encoder-workbench must exist in redhat-ods-applications.

        The RHOAI dashboard only discovers notebook images from ImageStreams in
        this namespace.  An ImageStream in any other namespace is invisible to
        the dashboard image selector.
        """
        assert resource_exists("imagestream", _IS_NAME, namespace=_RHOAI_NAMESPACE), (
            f"ImageStream {_IS_NAME!r} not found in namespace {_RHOAI_NAMESPACE!r}. "
            "The RHOAI dashboard only discovers notebook images from ImageStreams in "
            f"{_RHOAI_NAMESPACE!r}.  Check that the GitOps ImageStream manifest "
            "sets metadata.namespace: redhat-ods-applications."
        )

    def test_imagestream_has_notebook_image_label(self) -> None:
        """ImageStream must carry label opendatahub.io/notebook-image=true.

        This label is the discovery mechanism — the RHOAI dashboard lists only
        ImageStreams in redhat-ods-applications that have this label.
        """
        is_ = oc_json(
            ["get", "imagestream", _IS_NAME],
            namespace=_RHOAI_NAMESPACE,
        )
        labels = is_["metadata"].get("labels", {})
        assert labels.get("opendatahub.io/notebook-image") == "true", (
            f"ImageStream {_IS_NAME!r} is missing label "
            "'opendatahub.io/notebook-image: true'.  "
            "Without this label the RHOAI dashboard will not discover the image."
        )

    def test_imagestream_spec_tags_not_empty(self) -> None:
        """ImageStream spec.tags must be non-empty.

        The RHOAI dashboard reads spec.tags (the declarative tag list) to
        populate the workbench image selector.  An ImageStream with spec.tags=[]
        shows as 'image deleted' even when the build has pushed to status.tags.
        status.tags alone is insufficient.
        """
        is_ = oc_json(
            ["get", "imagestream", _IS_NAME],
            namespace=_RHOAI_NAMESPACE,
        )
        spec_tags = is_.get("spec", {}).get("tags", [])
        assert spec_tags, (
            f"ImageStream {_IS_NAME!r} has no spec.tags entries.  "
            "The RHOAI dashboard reads spec.tags to populate the image selector — "
            "status.tags alone is not sufficient.  "
            "Add a spec.tags entry for 'latest' with from.kind=DockerImage pointing "
            "to the built image reference."
        )

    def test_imagestream_latest_tag_in_spec(self) -> None:
        """spec.tags must include a 'latest' entry with a from source.

        The Notebook CR annotation last-image-selection references
        'pragma-encoder-workbench:latest'.  The corresponding spec.tags entry
        must declare where 'latest' resolves to, otherwise the dashboard cannot
        show the image as selectable.
        """
        is_ = oc_json(
            ["get", "imagestream", _IS_NAME],
            namespace=_RHOAI_NAMESPACE,
        )
        spec_tags = is_.get("spec", {}).get("tags", [])
        tag_names = [t.get("name") for t in spec_tags]
        assert _EXPECTED_TAG in tag_names, (
            f"ImageStream {_IS_NAME!r} spec.tags does not contain a "
            f"{_EXPECTED_TAG!r} entry.  "
            f"Found tags: {tag_names}.  "
            "Add spec.tags[name=latest] with from.kind=DockerImage."
        )
        latest_tag = next(t for t in spec_tags if t.get("name") == _EXPECTED_TAG)
        from_ = latest_tag.get("from", {})
        assert from_.get("kind") and from_.get("name"), (
            f"spec.tags[{_EXPECTED_TAG!r}].from is missing kind or name.  "
            f"Got: {from_!r}.  "
            "The from field must point to the built image: "
            "from.kind=DockerImage, from.name=<image-registry-ref>."
        )

    def test_imagestream_latest_tag_has_software_annotation(self) -> None:
        """spec.tags[latest] must have opendatahub.io/notebook-software annotation.

        This annotation populates the software badges displayed alongside the
        image name in the dashboard image selector.  Without it the entry shows
        no version information and may be harder to identify.
        """
        is_ = oc_json(
            ["get", "imagestream", _IS_NAME],
            namespace=_RHOAI_NAMESPACE,
        )
        spec_tags = is_.get("spec", {}).get("tags", [])
        latest_tag = next(
            (t for t in spec_tags if t.get("name") == _EXPECTED_TAG), None
        )
        if latest_tag is None:
            pytest.skip(f"spec.tags[{_EXPECTED_TAG!r}] not present — covered by prior test")

        ann = latest_tag.get("annotations", {})
        assert "opendatahub.io/notebook-software" in ann, (
            f"spec.tags[{_EXPECTED_TAG!r}] is missing annotation "
            "'opendatahub.io/notebook-software'.  "
            "Add a JSON array of {name, version} objects to populate the "
            "software badges in the RHOAI dashboard image selector."
        )

    def test_imagestream_latest_tag_is_recommended(self) -> None:
        """spec.tags[latest] must have opendatahub.io/workbench-image-recommended=true.

        This annotation marks the tag as the recommended/current version in the
        RHOAI dashboard, preventing it from appearing as an outdated tag.
        """
        is_ = oc_json(
            ["get", "imagestream", _IS_NAME],
            namespace=_RHOAI_NAMESPACE,
        )
        spec_tags = is_.get("spec", {}).get("tags", [])
        latest_tag = next(
            (t for t in spec_tags if t.get("name") == _EXPECTED_TAG), None
        )
        if latest_tag is None:
            pytest.skip(f"spec.tags[{_EXPECTED_TAG!r}] not present — covered by prior test")

        ann = latest_tag.get("annotations", {})
        assert ann.get("opendatahub.io/workbench-image-recommended") == "true", (
            f"spec.tags[{_EXPECTED_TAG!r}] annotation "
            "'opendatahub.io/workbench-image-recommended' is not 'true'.  "
            f"Got: {ann.get('opendatahub.io/workbench-image-recommended')!r}.  "
            "Without this, the tag may appear as outdated in the dashboard."
        )


# ---------------------------------------------------------------------------
# TestHardwareProfileDashboardContract
# ---------------------------------------------------------------------------


class TestHardwareProfileDashboardContract:
    """HardwareProfile must be correctly shaped for RHOAI dashboard visibility.

    The RHOAI dashboard lists HardwareProfiles from redhat-ods-applications.
    Two conditions make a profile invisible:
      - opendatahub.io/disabled: "true"
      - opendatahub.io/dashboard-feature-visibility annotation present with
        an empty array '[]' — observed to suppress the profile from the selector.
    """

    def test_hardware_profile_exists(self) -> None:
        """HardwareProfile pragma-encoder-gpu must exist in redhat-ods-applications.

        Deployed by GitOps.  Without this profile the workbench shows
        'no matching hardware profile' in the dashboard and the Notebook CR
        has no valid hardware binding.
        """
        assert resource_exists(
            "hardwareprofile.infrastructure.opendatahub.io",
            _HP_NAME,
            namespace=_RHOAI_NAMESPACE,
        ), (
            f"HardwareProfile {_HP_NAME!r} not found in {_RHOAI_NAMESPACE!r}.  "
            "Verify GitOps has synced openshift/gitops/notebook-image/hardware-profile-gpu.yaml."
        )

    def test_hardware_profile_not_disabled(self) -> None:
        """HardwareProfile must not have opendatahub.io/disabled=true.

        A disabled profile is hidden from the dashboard hardware selector.
        """
        hp = oc_json(
            ["get", "hardwareprofile.infrastructure.opendatahub.io", _HP_NAME],
            namespace=_RHOAI_NAMESPACE,
        )
        disabled = hp["metadata"].get("annotations", {}).get(
            "opendatahub.io/disabled", "false"
        )
        assert disabled != "true", (
            f"HardwareProfile {_HP_NAME!r} has opendatahub.io/disabled=true.  "
            "Set this annotation to 'false' or remove it to make the profile visible."
        )

    def test_hardware_profile_no_feature_visibility_annotation(self) -> None:
        """HardwareProfile must NOT have opendatahub.io/dashboard-feature-visibility.

        An empty array value '[]' for this annotation was observed to suppress
        the profile from the RHOAI dashboard hardware selector entirely.
        The annotation must be absent for global visibility.
        """
        hp = oc_json(
            ["get", "hardwareprofile.infrastructure.opendatahub.io", _HP_NAME],
            namespace=_RHOAI_NAMESPACE,
        )
        ann = hp["metadata"].get("annotations", {})
        assert "opendatahub.io/dashboard-feature-visibility" not in ann, (
            f"HardwareProfile {_HP_NAME!r} has annotation "
            "'opendatahub.io/dashboard-feature-visibility' = "
            f"{ann['opendatahub.io/dashboard-feature-visibility']!r}.  "
            "An empty array '[]' suppresses the profile from the dashboard selector.  "
            "Remove this annotation entirely from hardware-profile-gpu.yaml."
        )

    def test_hardware_profile_has_gpu_identifier(self) -> None:
        """HardwareProfile must declare an Accelerator identifier for nvidia.com/gpu.

        This identifier drives the resource request/limit injected into the
        workbench pod.  Without it the profile appears as CPU-only in the
        dashboard hardware selector.
        """
        hp = oc_json(
            ["get", "hardwareprofile.infrastructure.opendatahub.io", _HP_NAME],
            namespace=_RHOAI_NAMESPACE,
        )
        identifiers = hp.get("spec", {}).get("identifiers", [])
        gpu_identifiers = [
            i for i in identifiers
            if i.get("identifier") == "nvidia.com/gpu"
            and i.get("resourceType") == "Accelerator"
        ]
        assert gpu_identifiers, (
            f"HardwareProfile {_HP_NAME!r} has no Accelerator identifier for "
            "'nvidia.com/gpu'.  "
            f"Found identifiers: {[i.get('identifier') for i in identifiers]}.  "
            "Add an identifier entry with resourceType=Accelerator and "
            "identifier=nvidia.com/gpu."
        )

    def test_hardware_profile_scheduling_type_node(self) -> None:
        """HardwareProfile spec.scheduling.type must be 'Node'.

        The infrastructure.opendatahub.io/v1 schema requires nodeSelector to
        live under spec.scheduling.node.nodeSelector, not at spec.nodeSelector.
        spec.scheduling.type=Node is required to activate node-based scheduling.
        """
        hp = oc_json(
            ["get", "hardwareprofile.infrastructure.opendatahub.io", _HP_NAME],
            namespace=_RHOAI_NAMESPACE,
        )
        scheduling_type = hp.get("spec", {}).get("scheduling", {}).get("type")
        assert scheduling_type == "Node", (
            f"HardwareProfile {_HP_NAME!r} spec.scheduling.type is {scheduling_type!r}, "
            "expected 'Node'.  "
            "Set spec.scheduling.type: Node and place nodeSelector under "
            "spec.scheduling.node.nodeSelector."
        )


# ---------------------------------------------------------------------------
# TestNotebookCRDashboardContract
# ---------------------------------------------------------------------------


class TestNotebookCRDashboardContract:
    """Notebook CR must carry the annotations and labels required by RHOAI 3.4.

    Missing labels suppress the dashboard Open button.
    Missing hardware-profile annotations cause 'migration required' warnings.
    A mismatched last-image-selection causes 'image deleted' in the detail view.
    """

    def test_notebook_cr_exists(self, test_namespace: str) -> None:
        """Notebook CR pragma-encoder-workbench must exist in PRAGMA_TEST_NAMESPACE."""
        assert resource_exists("notebook", _NB_NAME, namespace=test_namespace), (
            f"Notebook {_NB_NAME!r} not found in namespace {test_namespace!r}.  "
            "Verify ArgoCD has synced openshift/gitops/workbench/notebook.yaml."
        )

    def test_notebook_cr_has_dashboard_labels(self, test_namespace: str) -> None:
        """Notebook CR must have both opendatahub.io/dashboard and odh-managed labels.

        These labels are required for the RHOAI dashboard to show the Open button.
        Without either label the workbench appears in the list but the button is
        permanently inactive.
        """
        nb = oc_json(
            ["get", "notebook", _NB_NAME],
            namespace=test_namespace,
        )
        labels = nb["metadata"].get("labels", {})
        assert labels.get("opendatahub.io/dashboard") == "true", (
            f"Notebook {_NB_NAME!r} is missing label 'opendatahub.io/dashboard: true'.  "
            "This label is required for the RHOAI dashboard Open button to be active."
        )
        assert labels.get("opendatahub.io/odh-managed") == "true", (
            f"Notebook {_NB_NAME!r} is missing label 'opendatahub.io/odh-managed: true'.  "
            "This label is required for the RHOAI dashboard Open button to be active."
        )

    def test_notebook_cr_has_inject_auth_annotation(self, test_namespace: str) -> None:
        """Notebook CR must use inject-auth (RHOAI 3.4) not inject-oauth (RHOAI 3.3).

        RHOAI 3.4 replaced the oauth-proxy sidecar with kube-rbac-proxy.
        The controller injects kube-rbac-proxy only when
        notebooks.opendatahub.io/inject-auth='true' is set.
        inject-oauth='true' on a 3.4 cluster does not inject any auth sidecar,
        which permanently disables the dashboard Open button.
        """
        nb = oc_json(
            ["get", "notebook", _NB_NAME],
            namespace=test_namespace,
        )
        ann = nb["metadata"].get("annotations", {})
        assert ann.get("notebooks.opendatahub.io/inject-auth") == "true", (
            f"Notebook {_NB_NAME!r} annotation 'notebooks.opendatahub.io/inject-auth' "
            f"is not 'true' (got {ann.get('notebooks.opendatahub.io/inject-auth')!r}).  "
            "RHOAI 3.4 requires inject-auth=true to inject the kube-rbac-proxy sidecar.  "
            "Using the old inject-oauth annotation on a 3.4 cluster disables the Open button."
        )
        assert "notebooks.opendatahub.io/inject-oauth" not in ann, (
            f"Notebook {_NB_NAME!r} still has the deprecated "
            "'notebooks.opendatahub.io/inject-oauth' annotation.  "
            "Remove it — it must not coexist with inject-auth on RHOAI 3.4."
        )

    def test_notebook_cr_hardware_profile_annotations(self, test_namespace: str) -> None:
        """Notebook CR must reference pragma-encoder-gpu in redhat-ods-applications.

        Without these annotations the dashboard shows 'migration required' and
        'no matching hardware profile'.
        """
        nb = oc_json(
            ["get", "notebook", _NB_NAME],
            namespace=test_namespace,
        )
        ann = nb["metadata"].get("annotations", {})
        assert ann.get("opendatahub.io/hardware-profile-name") == _HP_NAME, (
            f"Notebook {_NB_NAME!r} annotation 'opendatahub.io/hardware-profile-name' "
            f"is {ann.get('opendatahub.io/hardware-profile-name')!r}, expected {_HP_NAME!r}.  "
            "Dashboard shows 'migration required' when this annotation is missing or wrong."
        )
        assert ann.get("opendatahub.io/hardware-profile-namespace") == _RHOAI_NAMESPACE, (
            f"Notebook {_NB_NAME!r} annotation 'opendatahub.io/hardware-profile-namespace' "
            f"is {ann.get('opendatahub.io/hardware-profile-namespace')!r}, "
            f"expected {_RHOAI_NAMESPACE!r}."
        )

    def test_notebook_cr_image_selection_matches_imagestream(
        self, test_namespace: str
    ) -> None:
        """last-image-selection must reference a tag present in the ImageStream spec.tags.

        If last-image-selection references a tag that is not in spec.tags the
        dashboard shows 'image deleted' even when the image is running fine.
        """
        nb = oc_json(
            ["get", "notebook", _NB_NAME],
            namespace=test_namespace,
        )
        ann = nb["metadata"].get("annotations", {})
        last_selection = ann.get("notebooks.opendatahub.io/last-image-selection", "")
        assert last_selection, (
            f"Notebook {_NB_NAME!r} is missing annotation "
            "'notebooks.opendatahub.io/last-image-selection'.  "
            "This annotation tells the dashboard which ImageStream tag is selected."
        )

        # last-image-selection format: "<imagestream-name>:<tag>"
        parts = last_selection.split(":")
        assert len(parts) == 2 and parts[0] and parts[1], (
            f"last-image-selection {last_selection!r} is not in '<name>:<tag>' format."
        )
        selected_tag = parts[1]

        # Verify the referenced tag exists in spec.tags
        is_ = oc_json(
            ["get", "imagestream", _IS_NAME],
            namespace=_RHOAI_NAMESPACE,
        )
        spec_tag_names = [t.get("name") for t in is_.get("spec", {}).get("tags", [])]
        assert selected_tag in spec_tag_names, (
            f"Notebook last-image-selection references tag {selected_tag!r} but "
            f"ImageStream {_IS_NAME!r} spec.tags only contains: {spec_tag_names}.  "
            "Dashboard shows 'image deleted' when the referenced tag is absent from spec.tags."
        )
