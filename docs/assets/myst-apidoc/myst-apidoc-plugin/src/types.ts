import type { DirectiveContext } from 'myst-common';

export type Parser = DirectiveContext['parseMyst'];

export type Parameter = {
  name: string; // For errors, name is just an empty string; the "type" is the name
  type: string;
  desc: string;
};

export type Func = {
  Summary?: string[]; // Trim then join with space is fine
  'Extended Summary'?: string | []; // String with new-line chars in it
  Parameters?: Parameter[];
  'Other Parameters'?: Parameter[];
  Returns?: Parameter[];
  Raises?: Parameter[];
  Warns?: Parameter[];
  Notes?: string | []; // String with new-line chars in it
  References?: string[]; // Could do something to split on '.. [1]' prefix or just pull out DOIs or something.
  Examples?: string[] | ''; // Mostly code and output, but sometimes narrative. Cannot just trim and join with space.
  'See Also'?: ([string, null] | string)[][][]; // Not sure about the structure, but if we filter and flat, the strings are <function> (in the same submodule) or <module.submodule.function> (in different submodules)

  // No examples to confirm the following types
  Signature?: string;
  Attributes?: Parameter[];
  Methods?: any[];
  Yields?: Parameter[];
  Receives?: Parameter[];
  Warnings?: Parameter[];
  index?: Record<string, never>;
};

export type Submodule = Record<string, Func>;

export type Module = Record<string, Submodule>;

export type Options = {
  depth: number;
  module?: string;
  submodule?: string;
  function?: string;
};
