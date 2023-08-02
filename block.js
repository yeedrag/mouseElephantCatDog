const dbg = (...x) => console.log(...x);

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

		source = {args: {}, ...source}

		// .type
		Object.defineProperty(this, "type", {value: source.type, enumerable: true});
			// by default: writable: false and enumerable: false
	
		// ... other source import should be done here
	}

	// .args
	#args = new Proxy({}, {
		set: (obj, key, value) => {
			// should be re-considered
			this.dispatchEvent(
				new CustomEvent("argChange", {
					detail: { argName: key } 
				})
			);
			obj[key] = value;
			return true;
		}
	});
	get args(){ return this.#args; }

	// .exportObj()
	exportObj(){
		return {
			type: this.type,
			args: this.args
		};
	}

	// .element and .*El
	#element = (()=>{
		const ele = document.createElement("div");
		ele.classList.add("block");
		ele.innerHTML = `
			<div class="inputPorts"></div>
			<div class="header">${"NOT IMPLEMENTED HERE"}</div>
			<div class="content">${"NOT IMPLEMENTED HERE"}</div>
			<div class="outputPorts"></div>
		`;

		const [inputPortsEl, headerEl, contentEl, outputPortsEl] = ele.children;

		// move by dragging
		headerEl.addEventListener("mousedown", e => {
			const move = e => {
				this.pos.x += e.movementX;
				this.pos.y += e.movementY;
			}
			document.body.addEventListener("mousemove", move);
			document.body.addEventListener("mouseup", e => {
				document.body.removeEventListener("mousemove", move);
			}, { once: true })
		});

		return ele;
	})();
	get element(){ return this.#element }
	get inputPortsEl(){ return this.#element?.querySelector(":scope > .inputPorts") }
	get header(){ return this.#element?.querySelector(":scope > .header") }
	get contentEl(){ return this.#element?.querySelector(":scope > .content") }
	get outputPortsEl(){ return this.#element?.querySelector(":scope > .outputPorts") }

	/* -- START OF for BlockManager -- */
	// ._assignManager(), for BlockManager
	_assignManager({manager, id}){
		[this.#manager, this.#id] = [manager, id];

		// ... to be continued
	}

	// ._unassignManager(), for BlockManager
	_unassignManager(){
		[...this.#inputLines, ...this.#outputLines].forEach(line => line.remove());
		this.#manager = this.#id = undefined;
	}

	// .manager, read-only
	#manager = undefined;
	get manager(){ return this.#manager }

	// .remove()
	remove(){ this.#manager?.removeBlockById(this.id) }

	// .id, read-only
	#id = null;
	get id(){ return this.#id }

	// .pos
	#pos = new Proxy({ x: 0, y: 0 }, {
		set: (target, key, value) => {
			if(!["0", "1", "x", "y"].includes(key)) 
				return false;

			if(key == "0" || key == "x")
				target.x = value;
			if(key == "1" || key == "y")
				target.y = value;

			this.element.style["translate"] = `${target.x}px ${target.y}px`;
			this.dispatchEvent(new CustomEvent("posChange"));
			return true;
		},
		get: (target, key) => {
			if(key == "x" || key == "0") return target.x;
			if(key == "y" || key == "1") return target.y;
		}
	}); // initialized in constructor()
	set pos(value){
		let has = x => Object.keys(value).includes(x);
		if(has("x") || has("0")) this.#pos.x = value.x ?? value[0];
		if(has("y") || has("1")) this.#pos.y = value.y ?? value[1];

		if(!has("x") && !has("y") && !has("0") && !has("1"))
			return false;
		return true;
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

	#outputLines = [];
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
		if(!blockManager instanceof BlockManger)
			throw "param blockManager should be an instance of BlockManager";

		this.#manager = blockManager;
		this.#manager.ws.appendChild(this.element);

		// the inputPort to the line is the outputPort to the input block;
		// vice versa.
		["inputPort", "outputPort"].forEach(port => {
			Object.defineProperty(this, port, { value: document.createElement("div") });
			this[port].classList.add("port");
			this[port].addEventListener("pointerdown", async e => {
				const ptrDownTime = new Date();
				await new Promise(res => this[port].addEventListener("pointerup", res, {once: true}));
				const ptrUpTime = new Date();

				if(ptrUpTime - ptrDownTime > 200)
					return; // not a 'click'
				this.remove();
			})
		});
	}

	// .manager
	#manager = undefined;
	get manager(){ return this.#manager }

	// .remove()
	remove() {
		this._setInputBlock();
		this._setOutputBlock();
		this.manager._removeLine(this);
		this.element.remove();
	}

	// .inputBlockId
	#inputBlockId = undefined;
	get inputBlockId() { return this.#inputBlockId }

	// .inputBlock
	get inputBlock() { return this.#manager.getBlocks()[this.#inputBlockId] }

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

	// .outputBlockId
	#outputBlockId = undefined;
	get outputBlockId() { return this.#outputBlockId }

	// .outputBlock
	get outputBlock() { return this.#manager.getBlocks()[this.#outputBlockId] }

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

	// .element
	#element = (()=>{
		const ele = document.createElementNS("http://www.w3.org/2000/svg", "svg");
			// don't use doc~.createElement("svg" or "svg:svg") cuz that only
			// creates an HTML element with that tag name. The result will be 
			// an instance of HTMLUnknownElement, not a SVGElement.
		ele.classList.add("lineContainer");
		return ele;
	})();

	get element(){ return this.#element }

	// .redraw()
	redraw() {
		this.element.innerHTML = "";
		if(!this.inputPort.checkVisibility() || !this.outputPort.checkVisibility())
			return;

		// some useful utilities
		const abs = x => x > 0 ? x : -x;
		const min = (...v) => Math.min(...v);
		const max = (...v) => Math.max(...v);

		// get the centre points of ports
		const [c1, c2] = [this.inputPort, this.outputPort].map(port => {
			const wsPosition = this.#manager.getBoundingClientRect();
			const portBox = port.getBoundingClientRect();

			const x = 
				(portBox.x + portBox.width / 2 - anchor.x)
				/ this.#manager.wsCfg.scale;
			const y =
				(portBox.y + portBox.height / 2 - anchor.x)
				/ this.#manager.wsCfg.scale;
			return {x, y}
		});

		// # start to set for this.element
		// here is the ideal area of this.element if line's width is 0
		const area = {
			up: min(c1.y, c2.y),
			down: max(c1.y, c2.y),
			left: min(c1.x, c2.x),
			right: max(c1.x, c2.x),
			padding: 5
				// line has it's width, so leave some space
				// ref: ./index.css, .lineContainer
		}

		// here is the real area of this.element, where 'o' means all or outer
		Object.assign(area, {
			oUp: area.up - area.padding,
			oDown: area.down + area.padding,
			oLeft: area.left - area.padding,
			oRight: area.right + area.padding
				// things will be o right~
		});

		// here is the positions of the line's two ends relative to the real
		// area, and the width and height of the real area
		Object.assign(area, {
			oC1: { x: c1.x - area.oLeft, y: c1.y - area.oUp },
			oC2: { x: c2.x - area.oLeft, y: c2.y - area.oUp },

			oWidth: oRight - oLeft,
			oHeight: oDown - oUp
		});

		// configure this.element
		Object.entires({
			"class": "lineContainer",
			"width": `${oWidth}`,
			"height": `${oHeight}`,
			"style": `
				position: absolute;
				translate: ${oLeft}px ${oTop}px;
			`
		}).forEach(([attrName, attrValue]) => {
			this.element.setAttribute(attrName, attrValue)
		});
		this.element.innerHTML =
			`<line
				x1="${area.oC1.x}"
				y1="${area.oC1.y}"
				x2="${area.oC2.x}"
				y2="${area.oC2.y}"
			/>`;
	}
}

class BlockManager extends EventTarget {
	constructor({ workspaceContainer, source = {} }) {
		super();

		// .#wsC 
		if(!(workspaceContainer instanceof HTMLElement))
			throw "please provide an element as workspaceContainer";
		this.#wsC = workspaceContainer;

		// .#ws
		let wsCandidate = [...this.#wsC.querySelectorAll(":scope > :is(div, main)")]
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

		// now apply .wsCfg onto .ws and .wsC
		Object.keys(this.wsCfg).forEach(key => this.wsCfg[key] = this.wsCfg[key]);

		// allow users to scale the workspace by scrolling
		// and move it by dragging
		this.#initWorkspace({ ws: this.ws, wsC: this.wsC, wsCfg: this.wsCfg });
	} // constructor()

	#initWorkspace({ ws, wsC, wsCfg }) {
		// move the workspace while mouse dragging
		wsC.addEventListener("mousedown", e => {
			if (e.target != wsC) return;
			// prevent unexpected workspace movement (e.g. while blocks being dragged)

			const move = e => { // mousemove event
				wsCfg.movementX += e.movementX;
				wsCfg.movementY += e.movementY;
			}

			document.body.addEventListener("mousemove", move);
			document.body.addEventListener(
				"mouseup",
				e => document.body.removeEventListener("mousemove", move),
				{ once: true }
			);
		});

		// scale the workspace while scrolling
		{
			let mousePosInWsC = [0, 0]; // mouse's position in workspace container
			wsC.addEventListener("mousemove", e => {
				let wsCBox = wsC.getBoundingClientRect();
				mousePosInWsC = [e.x - wsCBox.x, e.y - wsCBox.y];
					// we can't just set that to [e.layerX, e.layerY] 'cuz
					// the "layer" may be any block the cursor's hovering on
			});

			wsC.addEventListener("wheel", e => {
				let oldScale = wsCfg.scale;

				wsCfg.scale *= 3 ** (e.deltaY * -0.001);
				let newScale = wsCfg.scale = Math.max(0.05, wsCfg.scale);

				// use mouse's pos as the scaling center
				let vec = [
					wsCfg.movementX - mousePosInWsC[0],
					wsCfg.movementY - mousePosInWsC[1],
				];

				let vecScale = newScale / oldScale;
				wsCfg.movementX = vec[0] * vecScale + mousePosInWsC[0];
				wsCfg.movementY = vec[1] * vecScale + mousePosInWsC[1];
			}, { passive: true });
		}
	}

	// .ws
	#ws = undefined;
	get ws() { return this.#ws }

	// .wsC
	#wsC = undefined;
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

			if (["movementX", "movementY", "showingX", "showingY"]) {
				// i.e., if movement changed

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

		// configure the Block
		block._assignManager({manager: this, id});
		this.#ws.appendChild(block.element);
		block.pos = [this.wsCfg.showingX, this.wsCfg.showingY];
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

	// .listLine()
	#lines = [];
	listLine(){ return [...this.#lines] }

	// .genLine()
	genLine(){
		let newLine = new Line(this);
			// Line() would add its element into this.ws
		this.#lines.push(newLine);
		return newLine;
	}

	// ._removeLine, for Line
	_removeLine(line) {
		this.#lines.splice(this.#lines.indexOf(line), 1);
	}
}

export { TypeManager, Block, Line, BlockManager }
