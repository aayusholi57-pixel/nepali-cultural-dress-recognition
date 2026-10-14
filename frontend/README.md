# Nepali Cultural Dress Recognition — Frontend

A responsive, standalone HTML/CSS/JavaScript interface for the existing model API. It is isolated in `frontend/` and does not replace model, dataset, training, Docker, API, or deployment files.

## Run locally

From the repository root, start a static server:

```powershell
py -m http.server 5500 --directory frontend
```

Open http://localhost:5500.

You can also open `index.html` directly, but a local static server is recommended.

## Connect the model API

1. Start or deploy the existing FastAPI application.
2. Enter the **exact prediction endpoint** in the Model API endpoint field.
3. Select an image and click **Analyze image**.
4. If the request fails, check the API route, request field name, response schema, and CORS settings.

The starter UI sends a multipart POST request with the file field named `file`. Confirm this matches your existing API implementation before using it. The frontend does not assume or change backend routes. If your API expects a different field name or JSON/base64 payload, adapt `frontend/app.js` to match it.

If frontend and API use different origins, configure FastAPI CORS for the frontend origin. Do not use wildcard origins with credentials in production.

## Safety and scope

- No backend or model files are changed by this frontend.
- No image is uploaded until the user clicks Analyze.
- Images are sent only to the endpoint entered by the user.
- The UI does not store uploaded images.
- Prediction field names are best-effort; inspect the raw response for your API's actual schema.
- Do not expose AWS credentials or other secrets in frontend code.

## Deployment

Deploy the `frontend/` directory as a static site (for example, with a static hosting provider). Configure the API URL in the UI and set appropriate backend CORS rules. The existing API deployment remains separate.
