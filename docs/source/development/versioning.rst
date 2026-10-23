Versioning and documentation releases
=====================================

Package versions
----------------

CPax uses calendar versioning: ``YYYY.M.N`` (for example,
``2026.10.0``). The corresponding annotated Git tag has a
``v`` prefix (``v2026.10.0``). A version is fixed by the tagged
source: a new release is not published by moving an existing tag.

The GitHub Actions workflows for preparing a CalVer release and publishing
that release to PyPI are separate, manually initiated processes. The
publishing step additionally requires validation and an approved GitHub
environment. See :doc:`publishing` for the authoritative procedure.

Read the Docs versions
----------------------

Read the Docs builds documentation for Git branches and tags:

* ``latest`` follows the default development branch, currently ``main``.
* ``stable`` follows the latest compatible stable version tag, when
  a suitable tag exists.
* Tagged documentation (for example ``v2026.10.0``) provides a
  version-specific reference for users of a published release.

The `Read the Docs version selector <https://docs.readthedocs.com/platform/stable/versions.html>`_
is provided by Read the Docs. We deliberately do not maintain an independent
version-switcher JSON file in the documentation theme.

Repository tags are not necessarily **active** in Read the Docs. After a
release, verify that the tag is discovered, activated, successfully built
and not hidden in the project's **Versions** dashboard; check that
``latest`` and ``stable`` point to the intended sources. If new tags
are not activated automatically, configure a suitable **Automation Rule**
for stable ``vYYYY.M.N`` release tags in the Read the Docs dashboard;
do not activate every temporary development branch.

These account-level settings cannot be changed by a repository pull
request. See the `Read the Docs versions guide
<https://docs.readthedocs.com/platform/stable/versions.html>`_ and
`automation rules guide
<https://docs.readthedocs.com/platform/stable/automation-rules.html>`_.
