import * as blkMgr from "./block.js";

// debugging management
const debugging = localStorage.getItem("debugging");
console.warn(
	debugging
? `If you are not a developer, don't paste any code here.
For Developers: run \`localStorage.setItem("debugging")\` and refresh the site to enable the debugging mode.`
: `You are in the debugging mode.`
)

// utils for debugging
if (debugging) globalThis.exposed = {};
globalThis.expose = debugging ? (obj) => {
	Object.assign(globalThis.exposed, obj);
} : () => {};
globalThis.dbg = debugging ? (...x) => console.log(...x) : () => {};

let typeManager = new blkMgr.TypeManager([
	{ name: "input" }
]);

let blockManager = new blkMgr.BlockManager({
	workspaceContainer: document.querySelector("#workspaceContainer")
});

const addBlock = (type) => {
	let initBlock = new blkMgr.Block({
		typeManager,
		source: {
			type,
			args: {}
		}
	});
	blockManager.addBlock(initBlock);
};

addBlock("input");
document.querySelector("button#addBlock").addEventListener("click", e => {
	const type = document.querySelector("#blockType").value;
	addBlock(type);
});

expose({ typeManager, blockManager, addBlock })
