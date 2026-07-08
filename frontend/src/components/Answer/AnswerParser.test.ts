import { cloneDeep } from 'lodash'

import { AskResponse, Citation } from '../../api' // Ensure this path matches the location of your types

import { enumerateCitations, parseAnswer, ParsedAnswer } from './AnswerParser' // Update the path accordingly

const sampleCitations: Citation[] = [
  {
    id: 'doc1',
    filepath: 'file1.pdf',
    part_index: undefined,
    content: '',
    title: null,
    url: null,
    metadata: null,
    chunk_id: null,
    reindex_id: null
  },
  {
    id: 'doc2',
    filepath: 'file1.pdf',
    part_index: undefined,
    content: '',
    title: null,
    url: null,
    metadata: null,
    chunk_id: null,
    reindex_id: null
  },
  {
    id: 'doc3',
    filepath: 'file2.pdf',
    part_index: undefined,
    content: '',
    title: null,
    url: null,
    metadata: null,
    chunk_id: null,
    reindex_id: null
  }
]

const sampleAnswer: AskResponse = {
  answer: 'This is an example answer with citations [doc1] and [doc2].',
  citations: cloneDeep(sampleCitations),
  generated_chart: null
}

describe('enumerateCitations', () => {
  it('assigns unique part_index based on filepath', () => {
    const results = enumerateCitations(cloneDeep(sampleCitations))
    expect(results[0].part_index).toEqual(1)
    expect(results[1].part_index).toEqual(2)
    expect(results[2].part_index).toEqual(1)
  })
})

const makeCitation = (id: string, url: string | null, filepath: string): Citation => ({
  id,
  filepath,
  part_index: undefined,
  content: '',
  title: null,
  url,
  metadata: null,
  chunk_id: null,
  reindex_id: null
})

describe('parseAnswer URL de-duplication (CB-0015)', () => {
  it('collapses two citations that share the same URL into one reference', () => {
    const answer: AskResponse = {
      answer: 'See the ordinance [doc1] and also [doc2].',
      citations: [makeCitation('doc1', 'https://burbank/ord-26-4038', 'ord.pdf'), makeCitation('doc2', 'https://burbank/ord-26-4038', 'ord.pdf')],
      generated_chart: null
    }
    const parsed = parseAnswer(answer) as NonNullable<ParsedAnswer>
    expect(parsed.citations).toHaveLength(1)
    expect(parsed.markdownFormatText).toContain('^1^')
    expect(parsed.markdownFormatText).not.toContain('^2^')
  })

  it('keeps distinct URLs as separate references', () => {
    const answer: AskResponse = {
      answer: 'First [doc1] then [doc2].',
      citations: [makeCitation('doc1', 'https://burbank/a', 'a.pdf'), makeCitation('doc2', 'https://burbank/b', 'b.pdf')],
      generated_chart: null
    }
    const parsed = parseAnswer(answer) as NonNullable<ParsedAnswer>
    expect(parsed.citations).toHaveLength(2)
    expect(parsed.markdownFormatText).toContain('^1^')
    expect(parsed.markdownFormatText).toContain('^2^')
  })

  it('does not collapse same-file chunks when URL is null (multi-part refs preserved)', () => {
    const answer: AskResponse = {
      answer: 'Part one [doc1] and part two [doc2].',
      citations: [makeCitation('doc1', null, 'file1.pdf'), makeCitation('doc2', null, 'file1.pdf')],
      generated_chart: null
    }
    const parsed = parseAnswer(answer) as NonNullable<ParsedAnswer>
    expect(parsed.citations).toHaveLength(2)
  })
})
