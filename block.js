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

		// set initial pos
		this.pos = [0, 0];
	}

	// .exportObj()
	exportObj(){
		return {
			type: this.type,
			args: this.args
		};
	}

	/* -- START OF BlockManager -- */
	// ._assignManager()
	_assignMmanager({manager, id, element, pos}){
		[this.#manager, this.#id, this.#element] = [manager, id, element];
		this.pos = pos || [0, 0];

		// now initialize the element
		this.element.classList
		// ... to be continued
	}

	// ._unassignManager()
	_unassignManager(){ this.#manager = this.#id = this.element = undefined; }

	// .remove()
	remove(){ this.parent.removeBlockById(this.id) }

	// .manager, read-only
	get manager(){ return this.#manager }

	// .id, read-only
	get id(){ return this.#id }

	// .pos
	set pos(value){
		this.dispatchEvent("posChange");
		this.pos = new Proxy({ x: value.x || value[0], y: value.y || value[1]}, {
			set: (target, key, value) => {
				target[key] = value;
				this.dispatchEvent("posChange");
				return true;
			}
		});
	}

	// ._linesToInput
	// .inputBlocks
	// ._linesFromOutput
	// .outputBlocks
	// ._exportObj()
	/* -- END OF For Block Manager -- */
}

class _Line {
	constructor(){
	}
}

class BlockManager extends EventTarget {
	constructor({ workspaceContainer, source = {} }) {
		super();

		// .#wsC
		this.#wsC = workspaceContainer || document.createElement("div");

		// .#ws
		let wsCandidate = [...this.#wsC.querySelectorAll(":scope > div")]
			.filter(ele => ele.children.length == 0)[0];
		if(wsCandidate){
			this.#ws = wsCandidate;
		}else{
			this.#ws = document.createElement("div");
			this.#wsC.appendChild(this.#ws);
		}

		/* to import from source */
		
	} // constructor()

	// .ws
	get ws() { return this.#ws }

	// .wsC
	get wsC() { return this.#wsC }

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
		block.assignManager({manager: this, id, element: elem});
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

	// .addLine
	// .#lines
	// .listLine
	// .removeLine
}

export {BlockManager}
