# mantis pi extension

This directory contains the `mantis.ts` [pi](https://pi.ai) extension that routes
selected user turns through the local mantis orchestrator (`:8088`). `/fugu` remains
an alias for `/mantis`.

## Files

- `mantis.ts` — extension entry point. Registers `/mantis off|trinity|conductor|auto` (and alias `/fugu`)
  and intercepts `input` events when the mode is not `off`.
- `tsconfig.json` — TypeScript configuration.
- `package.json` — minimal Node manifest for type checking.

## Build / type check

```bash
cd extensions
npm install
npm run typecheck
```

`mantis.ts` is loaded directly by pi as a TypeScript extension, so there is no
compile step for normal use.

## Test path

1. Start the local stack (`docker compose up -d` or `./scripts/run_mantis_native.sh`).
2. Copy or symlink `mantis.ts` into your pi extensions directory.
3. In a pi session run `/mantis auto` and send a coding prompt.
4. Check `~/.config/mantis/routing-log.jsonl` for routing decisions.
