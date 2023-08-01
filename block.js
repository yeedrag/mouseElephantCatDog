const clone = obj => {
	if(obj instanceof Array){
		return obj.map(ele => clone(ele))
	}else if(obj instanceof Object){
		return Object.fromEntries(Object.entries(obj).map(([key, value]) => [key, clone(value)]))
	}else if(obj instanceof String){
		return obj.toString()
	}else if(['string', 'number', 'function'].includes(typeof obj)){
		return obj // unlike an object, things in a function won't change, so we just reuse it
	}
};

class TypeManager {
	#registeredTypes = {};
	constructor(typeDefs) {
		Object.entries(typeDefs).forEach(([name, spec]) => {
			Object.defineProperty(this.#registeredTypes, name, { value: clone(spec) });
		});
	}
	addType(name, spec) {
		Object.defineProperty(this.#registeredTypes, name, { value: clone(spec) });
	}
	getType(name) {
		return clone(this.#registeredTypes[name])
	}
	listTypes() {
		return clone(this.#registeredTypes)
	}
}

class Block extends EventTarget {
	constructor({typeManager, source}){
		super();

		// .type
		Object.defineProperty(this, "type", {value: source.type, enumerable: true});
			// by default: writable: false and enumerable: false

		// .args
		this.args = new Proxy(source.args, {
			set: (obj, key, value) => {
				this.dispatchEvent(
					new CustomEvent("argChange", {
						detail: { argName: key } 
					})
				);
				obj[key] = value;
				return true;
			}
		});

		// set initial .pos
		this.pos = [0, 0];
	}

	// .exportObj()
	exportObj(){
		return {
			type: this.type,
			args: this.args
		};
	}

	/* -- START OF for BlockManager -- */
	// ._assignManager(), for BlockManager
	_assignManager({manager, id, element}){
		[this.#manager, this.#id, this.#element] = [manager, id, element];
		this.pos = [0, 0];

		// now initialize the element
		// check .*El
		this.#element.classList.add("block");
		this.#element.innerHTML = `
			<div class="inputPorts"></div>
			<div class="header">${header || ""}</div>
			<div class="content">${content || ""}</div>
			<div class="outputPorts"></div>
		`;

		// ... to be continued
	}

	// ._unassignManager(), for BlockManager
	_unassignManager(){ this.#manager = this.#id = this.element = undefined; }

	// .manager, read-only
	get manager(){ return this.#manager }

	// .element and .*El
	// check this.#element.innerHTML in ._assignManager
	get element(){ return this.#element }
	get inputPortsEl(){ return this.#element?.querySelector(":scope > .inputPorts") }
	get header(){ return this.#element?.querySelector(":scope > .header") }
	get contentEl(){ return this.#element?.querySelector(":scope > .content") }
	get outputPortsEl(){ return this.#element?.querySelector(":scope > .outputPorts") }

	// .remove()
	remove(){ this.#manager?.removeBlockById(this.id) }


	// .id, read-only
	get id(){ return this.#id }

	// .pos
	set pos(value){
		this.dispatchEvent("posChange");
		this.#pos = new Proxy({ x: value.x || value[0], y: value.y || value[1]}, {
			set: (target, key, value) => {
				target[key] = value;
				this.dispatchEvent("posChange");
				return true;
			}
		});
	}
	get pos(){ return this.#pos }
	
	/* -- START OF for Line -- */
	#inputLines = [];

	// ._assignInputLine(), for Line
	_assignInputLine(line, index=this.#inputLines.length) {
		this.inputPortsEl.insertBefore(
			line.outputPort,
			this.inputPortsEl.children[index] || null
				// insertBefore(): when reference node is null, insert at the end
		);
		this.#inputLines.splice(index, 0, line);
	}

	// ._unassignInputLine(), for Line
	_unassignInputLine(line){
		let index = this.#inputLines.indexOf(line);
		this.#inputLines.splice(index, 1);
		line.outputPort.remove();
	}

	// ._assignOnputLine(), for Line
	_assignOnputLine(line, index=this.#outputLines.length) {
		this.outputPortsEl.insertBefore(
			line.inputPort,
			this.outputPortsEl.children[index] || null
				// insertBefore(): when reference node is null, insert at the end
		);
		this.#outputLines.splice(index, 0, line);
	}

	// ._unassignOnputLine(), for Line
	_unassignOnputLine(line){
		let index = this.#outputLines.indexOf(line);
		this.#outputLines.splice(index, 1);
		line.inputPort.remove();
	}

	// ._exportObjForManager(), for BlockManager
	/* -- END OF for Line -- */
	/* -- END OF for Block Manager -- */
}

class Line {
	constructor(blockManager){
		this.#manager = blockManager;

		// the inputPort to the line is the outputPort to the input block;
		// vice versa.
		["inputPort", "outputPort"].forEach(port => {
			Object.defineProperty(this, port, { value: document.createElement("div") });
			this[port].classList.add("port");
			this[port].addEventListener("pointerdown", e => {
				const ptrDownTime = new Date();
				await new Promise(res => this[port].addEventListener("pointerup", res, {once: true}));
				const ptrUpTime = new Date();

				if(ptrUpTime - ptrDownTime > 200)
					return; // not a 'click'
				this.remove();
			})
		});

		this.#element = document.createElementNS("http://www.w3.org/2000/svg", "svg");
			// don't use doc~.createElement("svg" or "svg:svg") cuz that only
			// creates an HTML element with that tag name. The result will be 
			// an instance of HTMLUnknownElement, not a SVGElement.
		
	}

	// .manager
	get manager(){ return this.#manager }

	// .remove()
	remove() {
		this._setInputBlock();
		this._setOutputBlock();
	}

	// ._setInputBlock(), for BlockManger
	_setInputBlock(blockId, index) {
		// remove the old block
		if(this.#inputBlockId)
			this.inputBlock._unassignOutputLine(this);

		// set the id and configure the line
		this.#inputBlockId = blockId;
		if(blockId)
			this.inputBlock._assignOutputLine(this, index);
	}

	// .inputBlock
	get inputBlock() { return this.#manager.getBlocks()[this.#inputBlockId] }

	// .inputBlockId
	get inputBlockId() { return this.#inputBlockId }

	// ._setOutputBlock, for BlockManager
	_setOutputBlock(blockId, index) {
		// remove the old block
		if(this.#outputBlockId)
			this.outputBlock._unassignInputLine(this);

		// set the id and configure the line
		this.#outputBlockId = blockId;
		if(blockId)
			this.outputBlock._assignInputLine(this, index);
	}

	// .outputBlock
	get outputBlock() { return this.#manager.getBlocks()[this.#outputBlockId] }

	// .outputBlockId
	get outputBlockId() { return this.#outputBlockId }

	// .element
	get element(){ return this.#element }

	// .redraw()
	redraw() {
		if(!this.inputPort.checkVisibility() || !this.outputPort.checkVisibility())
			return;
		// should be implemented from the old code...
	}
}

class BlockManager extends EventTarget {
	constructor({ workspaceContainer, source = {} }) {
		super();

		// .#wsC
		this.#wsC = workspaceContainer || document.createElement("div");

		// .#ws
		let wsCandidate = [...this.#wsC.querySelectorAll(":scope > div")]
			.filter(ele => ele.children.length == 0)
			.sort((e1, e2) => e1.classList.has("workspace") ? -1 : 1)[0];
		if(wsCandidate){
			this.#ws = wsCandidate;
		}else{
			this.#ws = document.createElement("div");
			this.#wsC.appendChild(this.#ws);
		}

		/* to import from source */	
		// ... to be continued
	} // constructor()

	// .ws
	get ws() { return this.#ws }

	// .wsC
	get wsC() { return this.#wsC }

	// .wsCfg
	#wsCfg = new Proxy(
		{ movementX: 0, movementY: 0, scale: 1 },
		{ set: (target, key, value) => {
			target[key] = value;

			if (key == "scale") {
				// apply scale
				this.ws.style["scale"] = `${target.scale}`;
			}

			if (["showingX", "showingY"].includes(key)) {
				// calc the actually movement
				target.movementX = -target.showingX * target.scale;
				target.movementY = -target.showingY * target.scale;
			}

			if (["movementX", "movementY", "scale"].includes(key)) {
				// recalc the showing point
				target.showingX = -target.movementX / target.scale;
				target.showingY = -target.movementY / target.scale;
			}

			if (["movementX", "movementY", "showingX", "showingY"]) { // i.e., if movement changed
				this.ws.style["translate"] = `${target.movementX}px ${target.movementY}px`;
				// note: the "translate"d amounts seem not to be affected by "scale"
			}

			return true; // set handler should return true if success.
		} })

	// .wsCfg
	get wsCfg(){ return this.#wsCfg }

	#blocks = [];

	// .addBlock()
	addBlock(block) {
		if(!block instanceof Block)
			throw "addBlock(block) can only accept a Block as the block argument.";

		// determine id by subtract 1 from the return value of Array.prototype.push(), which is the new length
		const id = this.#blocks.push(block) - 1;

		// create element
		const elem = document.createElement("div");
		this.#ws.appendChild(elem);
		// ... the main parts are configured in Block

		// configure the Block
		block._assignManager({manager: this, id, element: elem});
	}

	// .blocks
	get blocks() { return [...this.#blocks] }

	// .removeBlockById()
	removeBlockById(id){
		if(['number', 'string'].includes(typeof id))
			throw "please specify the id";

		this.#blocks[id].element.remove();
		this.#blocks[id]._unassignManager();
		delete this.#blocks[id];
	}

	// .#lines
	// .addLine()
	// .listLine()
	// .removeLine()
}

export {BlockManager}
