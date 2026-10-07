// Kompilasi AuditTrail.sol memakai solcjs (npm), tanpa unduh binary eksternal.
// Output: build/AuditTrail.json  { abi, bytecode }
const fs = require("fs");
const path = require("path");
const solc = require("solc");

const src = fs.readFileSync(path.join(__dirname, "contracts", "AuditTrail.sol"), "utf8");
const input = {
  language: "Solidity",
  sources: { "AuditTrail.sol": { content: src } },
  settings: {
    optimizer: { enabled: true, runs: 200 },
    evmVersion: "paris",
    outputSelection: { "*": { "*": ["abi", "evm.bytecode.object"] } },
  },
};
const out = JSON.parse(solc.compile(JSON.stringify(input)));
const errs = (out.errors || []).filter((e) => e.severity === "error");
(out.errors || []).forEach((e) => console.error(e.formattedMessage));
if (errs.length) process.exit(1);
const c = out.contracts["AuditTrail.sol"]["AuditTrail"];
fs.mkdirSync(path.join(__dirname, "build"), { recursive: true });
fs.writeFileSync(
  path.join(__dirname, "build", "AuditTrail.json"),
  JSON.stringify({ abi: c.abi, bytecode: "0x" + c.evm.bytecode.object }, null, 2)
);
console.log("OK: build/AuditTrail.json (", c.evm.bytecode.object.length / 2, "bytes )");
