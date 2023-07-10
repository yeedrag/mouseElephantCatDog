const clone = obj => JSON.parse(JSON.stringify(obj));

class Block extends EventTarget {
	// https://developer.mozilla.org/en-US/docs/Web/API/EventTarget/EventTarget

	constructor (
		{ type = "customType", position = [0, 0], input = [], output = [], fields = {} }
	) {
		#data = { position, input, output, fields };
		#data.entries().forEach(([key, value]) => {
			if(typeof value != "object")
				throw new TypeError(
					`Invalid initial value: the type of the property ${key} is incorrect.`
				);
			#data[key] = new Proxy(value, {
				set: (t, tKey, tValue) => { // t for target
					this.dispatchEvent(new CustomEvent(
						`${key}Change`,
						{ key: tKey, oldValue: t[tKey], newValue: t[tKey] = tValue}
					));
					return true;
				}
			})
		});

		#data.type = type;
			// thanksfully, string is not an object
		#data = new Proxy(
			#data,
			{
				set: (target, key, value) => {
					if(key == "type") {
						throw new TypeError(
							`Invalid assignment: ` +
							`the property "type" shouldn't be changed after the Block created.`
						);
						return false;
					}

					this.dispatchEvent(new CustomEvent(
						`${key}Change`, 
						{ key: null, oldValue: target[key], newValue: value }
					));

					target[key] = value;
					return true;
				}
			}
		);
		get data() {
			return clone(this.#data);
		}
	}

	get data() {
		return clone(this.#data);
	}
	
	set data() {
		throw new TypeError(`Invalid assignment to read-only property "data"`);
		return false;
	}
} 

export {Block};
