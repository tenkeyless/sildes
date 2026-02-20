<!--
  제목·부제(선택) + 이미지(꽉 채움) + 캡션(선택, 있으면 한두 줄)

  이미지 채움 기준: 슬롯 이름으로 지정
  - ::img:: 또는 ::img-fit-width::  → 가로 기준 꽉 채움
  - ::img-fit-height::             → 세로 기준 꽉 채움

  캡션 없으면 이미지 영역이 더 넓게 잡힘.
-->

<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, useSlots, watch } from 'vue'

const props = defineProps({
  class: {
    type: String,
  },
  layoutClass: {
    type: String,
  },
})

const slots = useSlots()

const hasCaption = computed(() => !!slots.caption)
const hasImgFitHeight = computed(() => !!slots['img-fit-height'])
const hasImgFitWidth = computed(() => !!slots['img-fit-width'])

const imgFitMode = computed(() => {
  if (hasImgFitHeight.value) return 'fit-height'
  if (hasImgFitWidth.value || slots.img) return 'fit-width'
  return 'fit-width'
})

const gridRows = computed(() => hasCaption.value ? 'auto 1fr auto' : 'auto 1fr')

const rowImgEl = ref<HTMLElement | null>(null)
const fitTargetEl = ref<HTMLElement | null>(null)
const imgScale = ref(1)

let ro: ResizeObserver | null = null
let mo: MutationObserver | null = null

function recalcScale() {
  const container = rowImgEl.value
  const target = fitTargetEl.value
  if (!container || !target) return

  const cw = container.clientWidth
  const ch = container.clientHeight
  const tw = target.scrollWidth
  const th = target.scrollHeight
  if (!cw || !ch || !tw || !th) {
    imgScale.value = 1
    return
  }

  const widthScale = cw / tw
  const heightScale = ch / th
  let scale = 1

  if (imgFitMode.value === 'fit-height') {
    scale = Math.min(heightScale, 1)
    if (tw * scale > cw)
      scale = Math.min(scale, widthScale)
  }
  else {
    scale = Math.min(widthScale, 1)
    if (th * scale > ch)
      scale = Math.min(scale, heightScale)
  }

  imgScale.value = Number.isFinite(scale) && scale > 0 ? scale : 1
}

watch(imgFitMode, async () => {
  await nextTick()
  recalcScale()
})

watch(hasCaption, async () => {
  await nextTick()
  recalcScale()
})

onMounted(async () => {
  await nextTick()
  recalcScale()

  ro = new ResizeObserver(() => recalcScale())
  if (rowImgEl.value) ro.observe(rowImgEl.value)
  if (fitTargetEl.value) ro.observe(fitTargetEl.value)

  if (rowImgEl.value) {
    mo = new MutationObserver(() => recalcScale())
    mo.observe(rowImgEl.value, { childList: true, subtree: true, characterData: true, attributes: true })
  }
})

onBeforeUnmount(() => {
  ro?.disconnect()
  mo?.disconnect()
})
</script>

<template>
  <div
    class="slidev-layout img-caption w-full h-full gap-2"
    :class="[props.layoutClass, { 'has-caption': hasCaption }]"
    :style="{ gridTemplateRows: gridRows }"
  >
    <div class="row-header shrink-0">
      <slot />
    </div>
    <div
      ref="rowImgEl"
      class="row-img flex items-center justify-center min-h-0"
      :class="[imgFitMode, props.class]"
    >
      <div
        ref="fitTargetEl"
        class="img-fit-target"
        :style="{ transform: `scale(${imgScale})` }"
      >
        <slot v-if="hasImgFitHeight" name="img-fit-height" />
        <slot v-else-if="hasImgFitWidth" name="img-fit-width" />
        <slot v-else name="img" />
      </div>
    </div>
    <div v-if="hasCaption" class="row-caption shrink-0 min-h-0 text-center text-sm opacity-80">
      <slot name="caption" />
    </div>
  </div>
</template>

<style scoped>
.slidev-layout.img-caption {
  display: grid;
  grid-template-columns: 1fr;
  width: 100%;
  height: 100%;
}

.row-img {
  width: 100%;
  height: 100%;
}

.img-fit-target {
  transform-origin: center center;
  will-change: transform;
  display: inline-flex;
  align-items: center;
  justify-content: center;
}
</style>
