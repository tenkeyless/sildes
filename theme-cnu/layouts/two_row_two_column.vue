<!--
  2행×2열 그리드 (제목 + 4칸)
  슬롯: default(제목), r1-c1, r1-c2, r2-c1, r2-c2

  각 셀은 flex column. 세로 정렬:
  - 기본: 위 정렬 (justify-start)
  - 가운데: 셀 내용을 <div class="my-auto">로 감싸기
  - 아래: <div class="mt-auto">로 감싸기

  Usage:

```md
---
layout: two_row_two_column
---

# 제목

::r1-c1::
1행 1열

::r1-c2::
1행 2열

::r2-c1::
2행 1열

::r2-c2::
2행 2열
```
-->

<script setup lang="ts">
const props = defineProps({
  class: {
    type: String,
  },
  layoutClass: {
    type: String,
  },
})
</script>

<template>
  <div class="slidev-layout two-row-two-column w-full h-full" :class="props.layoutClass">
    <div class="row-header">
      <slot />
    </div>
    <div class="cell cell-r1-c1" :class="props.class">
      <slot name="r1-c1" />
    </div>
    <div class="cell cell-r1-c2" :class="props.class">
      <slot name="r1-c2" />
    </div>
    <div class="cell cell-r2-c1" :class="props.class">
      <slot name="r2-c1" />
    </div>
    <div class="cell cell-r2-c2" :class="props.class">
      <slot name="r2-c2" />
    </div>
  </div>
</template>

<style scoped>
.two-row-two-column {
  display: grid;
  grid-template-columns: 1fr 1fr;
  grid-template-rows: auto 1fr 1fr;
  column-gap: 1.5rem;
  row-gap: 0.75rem;
}

.row-header {
  grid-column: 1 / -1;
}

.cell {
  display: flex;
  flex-direction: column;
  justify-content: flex-start;
  align-items: stretch;
  min-height: 0;
  overflow: auto;
}

.cell-r1-c1 { grid-area: 2 / 1 / 3 / 2; }
.cell-r1-c2 { grid-area: 2 / 2 / 3 / 3; }
.cell-r2-c1 { grid-area: 3 / 1 / 4 / 2; }
.cell-r2-c2 { grid-area: 3 / 2 / 4 / 3; }
</style>
