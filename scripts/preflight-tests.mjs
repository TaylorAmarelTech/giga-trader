#!/usr/bin/env node
/**
 * Preflight Tests for Giga Trader Development Environment
 * Tests: Moltbot, Claude Code Harness, and project configuration
 */

import { execSync, spawn } from 'child_process';
import { existsSync, readFileSync } from 'fs';
import { join } from 'path';

const MOLTBOT_DIR = 'C:/Users/amare/moltbot';
const HARNESS_DIR = 'C:/Users/amare/claude-plugins/claude-code-harness';
const PROJECT_DIR = 'C:/Users/amare/OneDrive/Documents/giga_trader';
const CONFIG_DIR = 'C:/Users/amare/.moltbot';

const results = [];

function test(name, fn) {
  try {
    const result = fn();
    results.push({ name, status: 'PASS', message: result });
    console.log(`✓ ${name}`);
    if (result) console.log(`  └─ ${result}`);
  } catch (error) {
    results.push({ name, status: 'FAIL', message: error.message });
    console.log(`✗ ${name}`);
    console.log(`  └─ ${error.message}`);
  }
}

function exec(cmd, options = {}) {
  return execSync(cmd, { encoding: 'utf8', ...options }).trim();
}

console.log('\n═══════════════════════════════════════════════════════════');
console.log('  PREFLIGHT TESTS - Giga Trader Development Environment');
console.log('═══════════════════════════════════════════════════════════\n');

// ─────────────────────────────────────────────────────────────
// Node.js Environment
// ─────────────────────────────────────────────────────────────
console.log('▸ Node.js Environment\n');

test('Node.js version >= 22', () => {
  const version = process.version;
  const major = parseInt(version.slice(1).split('.')[0]);
  if (major < 22) throw new Error(`Node.js ${version} < v22 required`);
  return version;
});

test('pnpm available', () => {
  const version = exec('npm exec pnpm -- --version');
  return `v${version}`;
});

// ─────────────────────────────────────────────────────────────
// Moltbot
// ─────────────────────────────────────────────────────────────
console.log('\n▸ Moltbot\n');

test('Moltbot directory exists', () => {
  if (!existsSync(MOLTBOT_DIR)) throw new Error('Moltbot not found');
  return MOLTBOT_DIR;
});

test('Moltbot dependencies installed', () => {
  const nodeModules = join(MOLTBOT_DIR, 'node_modules');
  if (!existsSync(nodeModules)) throw new Error('node_modules missing');
  return 'node_modules present';
});

test('Moltbot dist built', () => {
  const dist = join(MOLTBOT_DIR, 'dist');
  if (!existsSync(dist)) throw new Error('dist folder missing - run build');
  return 'dist folder present';
});

test('Moltbot CLI executable', () => {
  const version = exec(`node ${join(MOLTBOT_DIR, 'moltbot.mjs')} --version 2>&1`);
  // Extract just the version number (last line typically)
  const versionLine = version.split('\n').pop();
  return versionLine;
});

test('Moltbot config exists', () => {
  const configPath = join(CONFIG_DIR, 'moltbot.json');
  if (!existsSync(configPath)) throw new Error('Config not found');
  const config = JSON.parse(readFileSync(configPath, 'utf8'));
  return `gateway.mode: ${config.gateway?.mode || 'unset'}`;
});

test('Moltbot gateway mode configured', () => {
  const configPath = join(CONFIG_DIR, 'moltbot.json');
  const config = JSON.parse(readFileSync(configPath, 'utf8'));
  if (!config.gateway?.mode) throw new Error('gateway.mode not set');
  return config.gateway.mode;
});

// ─────────────────────────────────────────────────────────────
// Claude Code Harness
// ─────────────────────────────────────────────────────────────
console.log('\n▸ Claude Code Harness\n');

test('Harness plugin directory exists', () => {
  if (!existsSync(HARNESS_DIR)) throw new Error('Harness not found');
  return HARNESS_DIR;
});

test('Harness commands available', () => {
  const commandsDir = join(HARNESS_DIR, 'commands');
  if (!existsSync(commandsDir)) throw new Error('commands folder missing');
  return 'commands folder present';
});

test('Harness agents available', () => {
  const agentsDir = join(HARNESS_DIR, 'agents');
  if (!existsSync(agentsDir)) throw new Error('agents folder missing');
  return 'agents folder present';
});

// ─────────────────────────────────────────────────────────────
// Project Configuration
// ─────────────────────────────────────────────────────────────
console.log('\n▸ Project Configuration\n');

test('Project directory exists', () => {
  if (!existsSync(PROJECT_DIR)) throw new Error('Project not found');
  return PROJECT_DIR;
});

test('CLAUDE.md exists', () => {
  const claudeMd = join(PROJECT_DIR, 'CLAUDE.md');
  if (!existsSync(claudeMd)) throw new Error('CLAUDE.md missing');
  return 'present';
});

test('Plans.md exists', () => {
  const plansMd = join(PROJECT_DIR, 'Plans.md');
  if (!existsSync(plansMd)) throw new Error('Plans.md missing');
  return 'present';
});

test('AGENTS.md exists', () => {
  const agentsMd = join(PROJECT_DIR, 'AGENTS.md');
  if (!existsSync(agentsMd)) throw new Error('AGENTS.md missing');
  return 'present';
});

test('Harness config exists', () => {
  const config = join(PROJECT_DIR, 'claude-code-harness.config.json');
  if (!existsSync(config)) throw new Error('Harness config missing');
  return 'present';
});

test('Git initialized', () => {
  const git = join(PROJECT_DIR, '.git');
  if (!existsSync(git)) throw new Error('.git missing');
  return 'initialized';
});

test('Source directory exists', () => {
  const src = join(PROJECT_DIR, 'src');
  if (!existsSync(src)) throw new Error('src/ missing');
  return 'present';
});

// ─────────────────────────────────────────────────────────────
// Summary
// ─────────────────────────────────────────────────────────────
console.log('\n═══════════════════════════════════════════════════════════');
const passed = results.filter(r => r.status === 'PASS').length;
const failed = results.filter(r => r.status === 'FAIL').length;
console.log(`  Results: ${passed} passed, ${failed} failed`);
console.log('═══════════════════════════════════════════════════════════\n');

if (failed > 0) {
  console.log('Failed tests:');
  results.filter(r => r.status === 'FAIL').forEach(r => {
    console.log(`  - ${r.name}: ${r.message}`);
  });
  console.log('');
  process.exit(1);
} else {
  console.log('All preflight checks passed! Ready for development.\n');
  process.exit(0);
}
