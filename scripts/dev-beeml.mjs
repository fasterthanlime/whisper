import { spawn } from "node:child_process";

const cwd = process.cwd();
const children = [];
let shuttingDown = false;

function color(label) {
  if (label === "beeml") return "\x1b[38;5;214m";
  return "\x1b[38;5;81m";
}

function prefixLines(stream, label, source) {
  const prefix = `${color(label)}[${label}:${source}]\x1b[0m `;
  let buffered = "";

  stream.on("data", (chunk) => {
    buffered += chunk.toString();
    const lines = buffered.split("\n");
    buffered = lines.pop() ?? "";
    for (const line of lines) {
      process[source].write(prefix + line + "\n");
    }
  });

  stream.on("end", () => {
    if (buffered.length > 0) {
      process[source].write(prefix + buffered + "\n");
      buffered = "";
    }
  });
}

function shutdown(signal = "SIGTERM") {
  if (shuttingDown) return;
  shuttingDown = true;
  for (const child of children) {
    if (!child.killed) {
      child.kill(signal);
    }
  }
}

function run(label, command, args) {
  const child = spawn(command, args, {
    cwd,
    env: process.env,
    stdio: ["inherit", "pipe", "pipe"],
  });
  children.push(child);
  prefixLines(child.stdout, label, "stdout");
  prefixLines(child.stderr, label, "stderr");

  child.on("exit", (code, signal) => {
    const status =
      signal != null ? `signal ${signal}` : `exit code ${code ?? 0}`;
    process.stderr.write(`${color(label)}[${label}:exit]\x1b[0m ${status}\n`);
    shutdown();
    process.exitCode = code ?? 0;
  });

  child.on("error", (error) => {
    process.stderr.write(
      `${color(label)}[${label}:error]\x1b[0m ${error.message}\n`,
    );
    shutdown();
    process.exitCode = 1;
  });
}

process.on("SIGINT", () => {
  shutdown("SIGINT");
});
process.on("SIGTERM", () => {
  shutdown("SIGTERM");
});

run("beeml", "zsh", [
  "-lc",
  [
    "source .envrc",
    "&&",
    "cargo watch",
    "-w rust",
    "-w data/phonetic-seed",
    "-x 'run -p beeml'",
  ].join(" "),
]);
run("web", "pnpm", ["--dir", "beeml-web", "dev"]);
