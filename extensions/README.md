# fugu pi extension

This directory contains the `fugu.ts` [pi](https://pi.ai) extension that routes
selected user turns through the local fugu orchestrator (`:8088`).

## Files

- `fugu.ts` — extension entry point. Registers `/fugu off|trinity|conductor|auto`
  and intercepts `input` events when the mode is not `off`.
- `tsconfig.json` — TypeScript configuration.
- `package.json` — minimal Node manifest for type checking.

## Build / type check

```bash
cd extensions
npm install
npm run typecheck
```

`fugu.ts` is loaded directly by pi as a TypeScript extension, so there is no
compile step for normal use.

## Test path

1. Start the local stack (`docker compose up -d` or `./scripts/run_openfugu_native.sh`).
2. Copy or symlink `fugu.ts` into your pi extensions directory.
3. In a pi session run `/fugu auto` and send a coding prompt.
4. Check `~/.config/fugu/routing-log.jsonl` for routing decisions.
