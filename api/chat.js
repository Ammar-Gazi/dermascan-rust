/**
 * api/chat.js — DermaScan AI · Gemini Proxy
 * ==========================================
 * Vercel serverless function that proxies requests to the Gemini API.
 * The API key lives ONLY in Vercel's encrypted environment variables
 * and is never exposed to the browser.
 *
 * Environment variable required (set in Vercel dashboard):
 *   GEMINI_API_KEY=AIzaSy...
 *
 * Endpoint: POST /api/chat
 * Body:    { history: [...], scanContext: "..." }
 * Returns: { reply: "..." } | { error: "..." }
 */

// ── Strict system prompt — identical to what was in the frontend ──
const SYSTEM_PROMPT = `You are DermaScan Assistant, an AI specialist embedded inside the DermaScan AI web application — a skin lesion analysis tool powered by ResNet18 trained on the HAM10000 dermoscopy dataset.

YOUR STRICT SCOPE — You ONLY answer questions about:
1. The 7 HAM10000 skin lesion classes: Melanocytic Nevi (NV), Melanoma (MEL), Benign Keratosis-like Lesions (BKL), Basal Cell Carcinoma (BCC), Actinic Keratoses (AKIEC), Vascular Lesions (VASC), Dermatofibroma (DF)
2. General skin lesion education: what each lesion looks like, risk level, typical appearance, when to seek care
3. How to interpret DermaScan AI results: confidence scores, softmax probabilities, what the predicted class means
4. Grad-CAM explainability: what the heatmap colors mean, why certain regions are highlighted, how to interpret activation maps
5. The HAM10000 dataset: what it is, how it was collected, class distribution
6. ResNet18 and transfer learning in skin cancer detection (conceptual, non-technical)
7. General guidance on when to consult a dermatologist
8. Skin lesion ABCDE rule (Asymmetry, Border, Color, Diameter, Evolution) as a general awareness tool

HARD RULES — You MUST follow these absolutely:
- NEVER invent, describe, or name any skin disease NOT in the 7 HAM10000 classes listed above
- NEVER provide a diagnosis, clinical opinion, or say what a specific user's lesion is
- NEVER give treatment advice, medication names, dosages, or medical procedures
- NEVER answer questions unrelated to skin lesions, dermatology, or this application
- NEVER speculate beyond what is scientifically established about these 7 lesion types
- If asked about anything outside scope, respond ONLY with: "I'm scoped specifically to DermaScan AI topics — skin lesion classes, result interpretation, and Grad-CAM. I can't help with that, but I'm happy to answer any DermaScan-related questions!"
- Always end answers about malignant/pre-cancerous lesions with a reminder to consult a licensed dermatologist
- Keep answers concise, factual, and helpful. Use bullet points for lists. Max 150 words per response.
- You are NOT a replacement for medical advice. State this clearly when relevant.
- Do not repeat the disclaimer in every single message — only when clinically relevant.

TONE: Professional, warm, precise. You are a knowledgeable assistant, not a doctor.`;

export default async function handler(req, res) {

  // ── CORS headers — restrict to your own Vercel domain in production ──
  // To lock down further, replace '*' with 'https://your-app.vercel.app'
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  // Handle preflight
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  // Only allow POST
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  // ── Read API key from environment — NEVER from client ──
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    console.error('GEMINI_API_KEY environment variable is not set');
    return res.status(500).json({ error: 'Server configuration error. API key not configured.' });
  }

  // ── Parse and validate request body ──
  let history, scanContext;
  try {
    ({ history = [], scanContext = null } = req.body);
  } catch {
    return res.status(400).json({ error: 'Invalid request body' });
  }

  if (!Array.isArray(history) || history.length === 0) {
    return res.status(400).json({ error: 'history must be a non-empty array' });
  }

  // ── Sanity-check history entries (prevent prompt injection via crafted payloads) ──
  const validRoles = new Set(['user', 'model']);
  const sanitizedHistory = history
    .filter(h => validRoles.has(h?.role) && typeof h?.parts?.[0]?.text === 'string')
    .map(h => ({
      role: h.role,
      parts: [{ text: String(h.parts[0].text).slice(0, 4000) }] // cap per-turn length
    }));

  if (sanitizedHistory.length === 0) {
    return res.status(400).json({ error: 'No valid history entries' });
  }

  // ── Build system instruction (inject scan context if present) ──
  const systemText = SYSTEM_PROMPT +
    (scanContext ? `\n\nCURRENT SCAN CONTEXT:\n${String(scanContext).slice(0, 500)}` : '');

  // ── Build Gemini request payload ──
  const payload = {
    system_instruction: { parts: [{ text: systemText }] },
    contents: sanitizedHistory,
    generationConfig: {
      maxOutputTokens: 400,
      temperature: 0.4,
      topP: 0.9,
    },
    safetySettings: [
      { category: 'HARM_CATEGORY_HARASSMENT',        threshold: 'BLOCK_MEDIUM_AND_ABOVE' },
      { category: 'HARM_CATEGORY_HATE_SPEECH',        threshold: 'BLOCK_MEDIUM_AND_ABOVE' },
      { category: 'HARM_CATEGORY_SEXUALLY_EXPLICIT',  threshold: 'BLOCK_MEDIUM_AND_ABOVE' },
      { category: 'HARM_CATEGORY_DANGEROUS_CONTENT',  threshold: 'BLOCK_MEDIUM_AND_ABOVE' },
    ],
  };

  // ── Call Gemini ──
  const endpoint = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${apiKey}`;

  try {
    const geminiRes = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    const data = await geminiRes.json();

    if (!geminiRes.ok) {
      const msg = data?.error?.message || `Gemini API error ${geminiRes.status}`;
      console.error('Gemini error:', msg);

      // Translate Gemini errors into clean client messages
      if (geminiRes.status === 400) return res.status(400).json({ error: 'Bad request to AI model.' });
      if (geminiRes.status === 429) return res.status(429).json({ error: 'Rate limit reached. Please wait and try again.' });
      if (geminiRes.status === 403) return res.status(403).json({ error: 'API quota exceeded or key restricted.' });
      return res.status(502).json({ error: 'AI model returned an error. Please try again.' });
    }

    const reply = data?.candidates?.[0]?.content?.parts?.[0]?.text;
    if (!reply) {
      // Might be blocked by safety filters
      const blockReason = data?.candidates?.[0]?.finishReason;
      if (blockReason === 'SAFETY') {
        return res.status(200).json({ reply: "I can't respond to that within my scope. Please ask about skin lesion classes, scan results, or Grad-CAM." });
      }
      return res.status(502).json({ error: 'Empty response from AI model.' });
    }

    return res.status(200).json({ reply });

  } catch (err) {
    console.error('Proxy fetch error:', err);
    return res.status(502).json({ error: 'Could not reach the AI model. Check your network.' });
  }
}
