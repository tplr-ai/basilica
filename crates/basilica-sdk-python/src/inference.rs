//! PyO3 bindings for `basilica_sdk::inference` (Managed Inference client).
//!
//! The Rust client is the single source of truth for the managed-inference
//! wire contract (endpoint resolution, query construction, error mapping).
//! This module only translates its typed values and errors into
//! Python-native shapes:
//!
//! - catalog models and usage rows become dicts, with `date` as
//!   `datetime.date` and `charge_credits` as `decimal.Decimal` (never float);
//! - `InferenceError` variants become the `basilica.exceptions` hierarchy,
//!   preserving structured attributes (`.cap`, `.retry_after`, `.model`,
//!   `.status_code`).
//!
//! The Python-side sugar (frozen dataclasses, auth resolution from the CLI
//! token store, `openai_client_args`, async wrappers) lives in
//! `python/basilica/inference.py`.

use basilica_sdk::inference::{
    InferenceClient as RustInferenceClient, InferenceError as RustInferenceError, InferenceModel,
    InferenceUsageRow, UsageQuery,
};
use chrono::{Datelike, NaiveDate};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDate, PyDict, PyModule};
#[cfg(feature = "stub-gen")]
use pyo3_stub_gen::derive::gen_stub_pyclass;
use std::time::Duration;
use tokio::runtime::Runtime;

/// Python wrapper for basilica_sdk::inference::InferenceClient
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass]
pub struct InferenceClient {
    inner: RustInferenceClient,
    runtime: Runtime,
}

#[pymethods]
impl InferenceClient {
    /// Create a new low-level inference client.
    ///
    /// Args:
    ///     api_key: Bearer credential (a `basilica_...` API key or OAuth JWT)
    ///     api_base: Management-plane base URL serving usage rollups
    ///         (defaults to the production basilica-api URL when None)
    ///     endpoint: Inference gateway endpoint override (defaults to
    ///         BASILICA_INFERENCE_ENDPOINT, then the production gateway)
    ///     timeout_secs: Request timeout in seconds (SDK default when None)
    #[new]
    #[pyo3(signature = (api_key, api_base=None, endpoint=None, timeout_secs=None))]
    fn new(
        py: Python<'_>,
        api_key: String,
        api_base: Option<String>,
        endpoint: Option<String>,
        timeout_secs: Option<u64>,
    ) -> PyResult<Self> {
        let runtime = Runtime::new()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?;

        let mut builder = RustInferenceClient::builder().with_api_key(&api_key);
        if let Some(api_base) = api_base {
            builder = builder.api_base(api_base);
        }
        if let Some(endpoint) = endpoint {
            builder = builder.endpoint(endpoint);
        }
        if let Some(timeout_secs) = timeout_secs {
            builder = builder.timeout(Duration::from_secs(timeout_secs));
        }

        let inner = builder.build().map_err(|e| map_inference_error(py, e))?;

        Ok(Self { inner, runtime })
    }

    /// The resolved inference gateway endpoint (no trailing slash)
    fn endpoint(&self) -> String {
        self.inner.endpoint().to_string()
    }

    /// The management-plane base URL the usage summary is fetched from
    fn api_base(&self) -> String {
        self.inner.api_base().to_string()
    }

    /// Base URL for OpenAI-compatible clients: `{endpoint}/v1`
    fn openai_base_url(&self) -> String {
        self.inner.openai_base_url()
    }

    /// List the tenant-visible model catalog (`GET {endpoint}/v1/models`)
    fn list_models(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        let client = self.inner.clone();

        let result = py.detach(|| {
            self.runtime
                .block_on(async move { client.list_models().await })
        });

        match result {
            Ok(models) => models.iter().map(|m| model_to_pydict(py, m)).collect(),
            Err(e) => Err(map_inference_error(py, e)),
        }
    }

    /// Fetch a single model's metadata (`GET {endpoint}/v1/models/{name}`)
    ///
    /// Raises the Python `InferenceModelNotFoundError` on 404.
    fn get_model(&self, py: Python<'_>, name: String) -> PyResult<Py<PyAny>> {
        let client = self.inner.clone();

        let result = py.detach(|| {
            self.runtime
                .block_on(async move { client.get_model(&name).await })
        });

        match result {
            Ok(model) => model_to_pydict(py, &model),
            Err(e) => Err(map_inference_error(py, e)),
        }
    }

    /// Fetch usage rollups from the management plane
    /// (`GET {api_base}/v1/inference/usage/summary`).
    ///
    /// Args:
    ///     from_date: First day to include, ISO "YYYY-MM-DD" (UTC, inclusive)
    ///     to_date: Last day to include, ISO "YYYY-MM-DD" (UTC, inclusive)
    ///     model: Restrict to a single model
    ///     kid: Restrict to a single API key ID
    #[pyo3(signature = (from_date=None, to_date=None, model=None, kid=None))]
    fn usage(
        &self,
        py: Python<'_>,
        from_date: Option<String>,
        to_date: Option<String>,
        model: Option<String>,
        kid: Option<String>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let query = UsageQuery {
            from: parse_iso_date(from_date)?,
            to: parse_iso_date(to_date)?,
            model,
            kid,
        };
        let client = self.inner.clone();

        let result = py.detach(|| {
            self.runtime
                .block_on(async move { client.usage(query).await })
        });

        match result {
            Ok(rows) => rows.iter().map(|r| usage_row_to_pydict(py, r)).collect(),
            Err(e) => Err(map_inference_error(py, e)),
        }
    }
}

/// Parse an optional ISO "YYYY-MM-DD" date filter
fn parse_iso_date(value: Option<String>) -> PyResult<Option<NaiveDate>> {
    value
        .map(|s| {
            NaiveDate::parse_from_str(&s, "%Y-%m-%d").map_err(|e| {
                PyValueError::new_err(format!("invalid date `{}` (expected YYYY-MM-DD): {}", s, e))
            })
        })
        .transpose()
}

/// Convert a catalog model to a Python dict mirroring the wire fields;
/// optionals absent on the wire are omitted from the dict.
fn model_to_pydict(py: Python<'_>, model: &InferenceModel) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("id", &model.id)?;
    if let Some(object) = &model.object {
        dict.set_item("object", object)?;
    }
    if let Some(created) = model.created {
        dict.set_item("created", created)?;
    }
    if let Some(owned_by) = &model.owned_by {
        dict.set_item("owned_by", owned_by)?;
    }
    Ok(dict.into_any().unbind())
}

/// Convert a usage row to a Python dict. `date` becomes a `datetime.date`
/// and `charge_credits` a `decimal.Decimal` parsed from the exact wire
/// string, so summing charges in Python never drifts through floats.
fn usage_row_to_pydict(py: Python<'_>, row: &InferenceUsageRow) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    let date = PyDate::new(
        py,
        row.date.year(),
        row.date.month() as u8,
        row.date.day() as u8,
    )?;
    dict.set_item("date", date)?;
    dict.set_item("model", &row.model)?;
    if let Some(tenant_id) = &row.tenant_id {
        dict.set_item("tenant_id", tenant_id)?;
    }
    if let Some(kid) = &row.kid {
        dict.set_item("kid", kid)?;
    }
    dict.set_item("prompt_tokens", row.prompt_tokens)?;
    dict.set_item("completion_tokens", row.completion_tokens)?;
    dict.set_item("cached_tokens", row.cached_tokens)?;
    let charge = PyModule::import(py, "decimal")?
        .getattr("Decimal")?
        .call1((row.charge_credits.to_string(),))?;
    dict.set_item("charge_credits", charge)?;
    Ok(dict.into_any().unbind())
}

/// Map a typed Rust [`RustInferenceError`] onto the `basilica.exceptions`
/// hierarchy, keeping the structured attributes the Python surface
/// documents (`.cap` / `.retry_after` on quota errors, `.model` on
/// not-found, `.status_code` where the contract assigns one).
fn map_inference_error(py: Python<'_>, error: RustInferenceError) -> PyErr {
    let message = error.to_string();
    match build_python_error(py, &error, &message) {
        Ok(err) => err,
        // If the exceptions module cannot be imported (unusual embedding),
        // still surface the underlying failure rather than the import error.
        Err(_) => PyRuntimeError::new_err(message),
    }
}

fn build_python_error(
    py: Python<'_>,
    error: &RustInferenceError,
    message: &str,
) -> PyResult<PyErr> {
    use basilica_sdk::inference::InferenceError as E;

    let exc = PyModule::import(py, "basilica.exceptions")?;
    let instance = match error {
        E::Auth => exc
            .getattr("InferenceAuthenticationError")?
            .call1((message,))?,
        E::Forbidden => {
            let kwargs = PyDict::new(py);
            kwargs.set_item("status_code", 403)?;
            exc.getattr("InferenceError")?
                .call((message,), Some(&kwargs))?
        }
        E::Invalid(_) => exc.getattr("InferenceError")?.call1((message,))?,
        E::ModelNotFound { model } => {
            let kwargs = PyDict::new(py);
            kwargs.set_item("model", model)?;
            exc.getattr("InferenceModelNotFoundError")?
                .call((message,), Some(&kwargs))?
        }
        E::InsufficientCredits => exc.getattr("InsufficientCreditsError")?.call1((message,))?,
        E::QuotaExceeded { cap, retry_after } => {
            let kwargs = PyDict::new(py);
            kwargs.set_item("cap", cap.as_str())?;
            if let Some(secs) = retry_after {
                kwargs.set_item("retry_after", *secs as f64)?;
            }
            exc.getattr("InferenceQuotaExceededError")?
                .call((message,), Some(&kwargs))?
        }
        E::Unavailable { retry_after } => {
            let kwargs = PyDict::new(py);
            if let Some(secs) = retry_after {
                kwargs.set_item("retry_after", *secs as f64)?;
            }
            exc.getattr("InferenceUnavailableError")?
                .call((message,), Some(&kwargs))?
        }
        E::Transport(_) => exc
            .getattr("NetworkError")?
            .call1((format!("Failed to reach inference endpoint: {}", message),))?,
        E::Decode(_) => {
            let kwargs = PyDict::new(py);
            kwargs.set_item("code", "INFERENCE_BAD_RESPONSE")?;
            exc.getattr("InferenceError")?
                .call((message,), Some(&kwargs))?
        }
    };
    Ok(PyErr::from_value(instance))
}
