import fs from 'node:fs';
import type {
  DirectiveData,
  DirectiveSpec,
  GenericNode,
  GenericParent,
  DirectiveContext,
} from 'myst-common';
import type { Func, Module, Options, Parameter, Parser, Submodule } from './types.js';
import type { VFile } from 'vfile';

export function optsToLabel(opts: Options) {
  const { module, submodule, function: func } = opts;
  const start = module ? `${module}${submodule ? '.' : ''}` : '';
  const middle = submodule ? `${submodule}${func ? '.' : ''}` : '';
  const end = func ? `${func}` : '';
  return `${start}${middle}${end}`;
}

export function parameterToMdast(param: Parameter, parse: Parser): GenericNode[] {
  const name = param.name || param.type; // Sometimes name is "" and type should be used for the name
  const type = param.name && param.type ? param.type : undefined;
  const term: GenericParent = {
    type: 'definitionTerm',
    children: [
      {
        type: 'text',
        value: name,
      },
    ],
  };
  if (type) {
    term.children.push(
      {
        type: 'text',
        value: ' : ',
      },
      {
        type: 'emphasis',
        children: [
          {
            type: 'text',
            value: type,
          },
        ],
      },
    );
  }
  return [
    term,
    {
      type: 'definitionDescription',
      children: parse(param.desc).children,
    },
  ];
}

export function parameterListToMdast(
  name: string,
  params: Parameter[],
  parse: Parser,
  opts: Options,
): GenericNode[] {
  if (params.length === 0) return [];
  return [
    {
      type: 'heading',
      depth: opts.depth,
      children: [
        {
          type: 'text',
          value: name,
        },
      ],
    },
    {
      type: 'definitionList',
      children: params.map((param) => parameterToMdast(param, parse)).flat(),
    },
  ];
}

export function functionToMdast(
  name: string,
  func: Func,
  parse: Parser,
  opts: Options,
): GenericNode[] {
  const newOpts = {
    depth: opts.depth + 1,
    module: opts.module,
    submodule: opts.submodule,
    function: name,
  };
  const section: GenericNode[] = [
    {
      type: 'mystTarget',
      label: optsToLabel(newOpts),
    },
    {
      type: 'heading',
      depth: opts.depth,
      children: [
        {
          type: 'text',
          value: name,
        },
      ],
    },
  ];
  if (func.Summary) {
    section.push(...parse(func.Summary.map((line) => line.trim()).join(' ')).children);
  }
  if (typeof func['Extended Summary'] === 'string') {
    section.push(...parse(func['Extended Summary']).children);
  }
  if (func.Parameters) {
    section.push(...parameterListToMdast('Parameters', func.Parameters, parse, newOpts));
  }
  if (func['Other Parameters']) {
    section.push(
      ...parameterListToMdast('Other Parameters', func['Other Parameters'], parse, newOpts),
    );
  }
  if (func.Returns) {
    section.push(...parameterListToMdast('Returns', func.Returns, parse, newOpts));
  }
  if (func.Raises) {
    section.push(...parameterListToMdast('Raises', func.Raises, parse, newOpts));
  }
  if (func.Warns) {
    section.push(...parameterListToMdast('Warns', func.Warns, parse, newOpts));
  }
  if (typeof func.Notes === 'string') {
    section.push(
      {
        type: 'heading',
        depth: opts.depth + 1,
        children: [
          {
            type: 'text',
            value: 'Notes',
          },
        ],
      },
      ...parse(func.Notes).children,
    );
  }
  if (func.References) {
    const referencesAST = {
      type: 'list',
      ordered: true,
      children: func.References
        .map(text => ({
          type: 'listItem',
          children: [{
            type: 'paragraph',
            children: parse(text).children
          }]
        }))
    };
    section.push(
      {
        type: 'heading',
        depth: opts.depth + 1,
        children: [
          {
            type: 'text',
            value: 'References',
          },
        ],
      },
      referencesAST
    );
  }
  if (func.Examples) {
    section.push(
      {
        type: 'heading',
        depth: opts.depth + 1,
        children: [
          {
            type: 'text',
            value: 'Examples',
          },
        ],
      },
      {
        type: 'code',
        lang: 'python',
        value: func.Examples.join('\n'),
      },
    );
  }
  if (func['See Also'] && func['See Also'].length > 0) {
    const seeAlso = func['See Also'].flat(2).filter((val) => val.length > 0);
    const seeAlsoXrefs = seeAlso
      .filter((val): val is [string, null] => typeof val !== 'string')
      .map(([val]) => val);
    const seeAlsoText = seeAlso.filter((val) => typeof val === 'string').join(' ');
    if (seeAlsoXrefs.length > 0 || seeAlsoText) {
      section.push({
        type: 'heading',
        depth: opts.depth + 1,
        children: [
          {
            type: 'text',
            value: 'See Also',
          },
        ],
      });
      if (seeAlsoXrefs.length > 0) {
        section.push({
          type: 'paragraph',
          children: seeAlsoXrefs
            .map((value, index) => {
              const xref = value.includes('.') ? value : optsToLabel({ ...opts, function: value });
              const children: GenericNode[] = [
                {
                  type: 'link',
                  url: `#${xref}`,
                  children: [
                    {
                      type: 'text',
                      value,
                    },
                  ],
                },
              ];
              if (index < seeAlso.length - 1) {
                children.push({
                  type: 'text',
                  value: ', ',
                });
              }
              return children;
            })
            .flat(),
        });
      }
      if (seeAlsoText) {
        section.push({
          type: 'paragraph',
          children: [
            {
              type: 'text',
              value: seeAlsoText,
            },
          ],
        });
      }
    }
  }
  return section;
}

export function submoduleToMdast(
  name: string,
  submodule: Submodule,
  parse: Parser,
  opts: Options,
): GenericNode[] {
  const newOpts = {
    depth: opts.depth + 1,
    module: opts.module,
    submodule: name,
  };
  const section: GenericNode[] = [
    {
      type: 'mystTarget',
      label: optsToLabel(newOpts),
    },
    {
      type: 'heading',
      depth: opts.depth,
      children: [
        {
          type: 'text',
          value: name,
        },
      ],
    },
  ];
  Object.entries(submodule).forEach(([funcName, func]) => {
    section.push(...functionToMdast(funcName, func, parse, newOpts));
  });
  return section;
}

export function moduleToMdast(module: Module, parse: Parser, opts: Options): GenericNode[] {
  const section: GenericNode[] = [];
  Object.entries(module).forEach(([submoduleName, submodule]) => {
    section.push(...submoduleToMdast(submoduleName, submodule, parse, opts));
  });
  return section;
}

export const apiDirective: DirectiveSpec = {
  name: 'apidoc',
  doc: 'Directive for loading docstrings (currently from fleece output)',
  arg: {
    type: String,
    doc: 'File with fleece output',
    required: true,
  },
  options: {
    module: {
      type: String,
      doc: 'Module name for cross-reference labels.',
    },
    depth: {
      type: Number,
      doc: 'Starting heading depth',
    },
  },
  run(data: DirectiveData, vfile: VFile, ctx: DirectiveContext) {
    const [filename, target] = (data.arg as string).split('#');
    const docJson = JSON.parse(fs.readFileSync(filename).toString());
    const opts = {
      depth: +(data.options?.depth ?? 1),
      module: data.options?.module ? (data.options.module as string) : undefined,
    };
    if (target) {
      const [submodule, func] = target.split('.');
      if (submodule && func && docJson[submodule]?.[func]) {
        return functionToMdast(func, docJson[submodule][func], ctx.parseMyst, {
          ...opts,
          submodule,
        });
      }
      if (submodule && docJson[submodule]) {
        return submoduleToMdast(submodule, docJson[submodule], ctx.parseMyst, opts);
      }
    }
    return moduleToMdast(docJson, ctx.parseMyst, opts);
  },
};
