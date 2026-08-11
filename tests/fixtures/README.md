# Test fixtures

Provenance matters here. A mapper tested against a payload someone imagined is
only tested against that imagination, so these are captured from the live APIs
wherever that was possible.

| File | Provenance |
|---|---|
| `gutendex_page1.json` | Captured from `GET https://gutendex.com/books?page=1` on 11 Aug 2026. Trimmed to the first 3 results; `next` rewritten to point at page 2 so the link walk is exercised end to end. |
| `gutendex_page2.json` | Captured from `?page=2` the same day. Trimmed to 2 results with `next: null` to terminate the walk. |
| `gutendex_malformed_items.json` | Derived from the captured page 1. One record has its title blanked and another has `id` removed, so per-item rejection can be tested without inventing a payload shape. |
| `openlibrary_search.json` | Captured from `GET https://openlibrary.org/search.json` on 11 Aug 2026 with the exact `fields` list the extractor sends. Per-work `isbn` arrays truncated to 6 entries; the live response carries 50+ across all editions. |
| `openlibrary_empty.json` | A zero-result response, hand-written — the shape is three scalar fields and an empty list. |
| `googlebooks_rate_limited.json` | **Real.** Captured on 11 Aug 2026, when the anonymous daily quota for this IP was exhausted. This is what a 429 from Google Books actually looks like. |
| `googlebooks_volumes.json` | **Not captured.** Written from Google's published Volume schema because the anonymous daily quota was exhausted on the day the extractor was built. Replace it with a captured payload once `PIPELINE_GOOGLEBOOKS_API_KEY` is configured — see the note below. |

## Replacing the Google Books fixture

```bash
curl -s "https://www.googleapis.com/books/v1/volumes?q=isbn:9780553380163&key=$KEY" \
  > tests/fixtures/googlebooks_volumes.json
```

The mapper tests should pass unchanged. If they do not, the hand-written fixture
was wrong about something and the captured one is right.
