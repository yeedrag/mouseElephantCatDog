import * as blkMgr from "./block.js";

let typeManager = new blkMgr.TypeManager({
	input: {}
});

let blockManager = new blkMgr.BlockManager({
	workspaceContainer: document.querySelector("#workspaceContainer")
});

{
	let initBlock = new blkMgr.Block({
		typeManager,
		source: {
			type: "input",
			args: {}
		}
	});
	blockManager.addBlock(initBlock);
}
