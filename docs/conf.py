# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from rocm_docs import ROCmDocs

external_projects_current_project = "hipsparselt"

docs_core = ROCmDocs("hipSPARSELt Documentation")
docs_core.run_doxygen(doxygen_root="doxygen", doxygen_path="doxygen/docBin/xml")
docs_core.enable_api_reference()
docs_core.setup()

extensions = ['sphinx_design', 'sphinx.ext.intersphinx']

exclude_patterns = ['reference/api-library.md']

external_toc_path = "./sphinx/_toc.yml"

suppress_warnings = ["etoc.toctree"]

for sphinx_var in ROCmDocs.SPHINX_VARS:
    globals()[sphinx_var] = getattr(docs_core, sphinx_var)
