<!--
  제목·부제 → 위쪽 그림 → 아래쪽 텍스트

  Usage:

```md
---
layout: top_img-bottom_text
---

# 제목

부제 (선택)

::top::

mermaid 또는 이미지

::bottom::

설명 텍스트/코드
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
  <div
    class="slidev-layout top-img-bottom-text w-full h-full grid grid-rows-[auto_1fr_1fr] gap-4"
    :class="props.layoutClass"
  >
    <div class="row-header shrink-0">
      <slot />
    </div>
    <div class="row-top flex items-center justify-center min-h-0" :class="props.class">
      <slot name="top" />
    </div>
    <div class="row-bottom min-h-0 overflow-auto" :class="props.class">
      <slot name="bottom" />
    </div>
  </div>
</template>
