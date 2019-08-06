<template>
  <v-hover v-slot:default="{ hover }">
	<v-card :class="{ 'pose-container': pose.id !== 7, 'pocket-box': pose.id === 7 }">

    <v-card-actions v-if="!(pose.id === 7)">
      <v-spacer />
      <v-btn icon @click="removePose()">
        <v-icon>clear</v-icon>
      </v-btn>
    </v-card-actions>
		<v-card-title v-if="!(pose.id === 7)">

			<div class="headline">
        <img :src="pose.image" class="pose">
        <v-expand-transition>
          <div
            v-if="hover"
            class="transition-fast-in-fast-out darken-2 pose-name"
          >
            {{ pose.name }}
          </div>
        </v-expand-transition>
      </div>
		</v-card-title>
		<v-container>
			<v-layout column>
        <Draggable
          v-model="draggableCards"
          :options="draggableOptions"
          style="min-height: 10px"
          class="pocket-container"
        >
          <v-flex
            v-for="card in draggableCards"
            :key="card.id"
            class="draggable-item"
          >
            <ActionCard :card="card"></ActionCard>
          </v-flex>
        </Draggable>
			</v-layout>
		</v-container>
	</v-card>
  </v-hover>
</template>

<script>
 import Draggable from 'vuedraggable'
 import ActionCard from '@/components/ActionCard'

  export default {
    components: {
      Draggable,
      ActionCard
    },
    props: {
      pose: {
        type: Object,
        required: true
      },
      cards: {
        type: Array,
        required: true
      }
    },
    computed: {
      draggableCards: {
        get() {
          return this.cards
        },
        set(newCards) {
          const payload = {
            cards: newCards,
            poseId: this.pose.id
          }
          this.$emit('set-cards', payload)
        }
      },
      draggableOptions() {
        return {
          group: {
            name: 'cards'
          },
          ghostClass: 'ghost'
        }
      }
    },
    methods: {
      removePose() {
        const payload = {
          poseId: this.pose.id
        }
        this.$emit('remove-pose', payload)
      }
    }
  }
</script>

<style lang="scss" scoped>
  .ghost {
    opacity: 0.4;
  }

  .draggable-item {
    cursor: pointer;
  }

  .headline {
    margin: 0 auto;
    position: relative;
  }
  // ポーズ画像関係
  .pose {
    width: 100%;
    height: 100%;
  }

  .pose-name {
    text-align: center;
    font-size: 20px;
    font-weight: bold;
    color: white;
    background-color: #E571C0;
    position: absolute;
    bottom: 0px;
    width: 100%;
  }

  // pocket
  .pocket-box {
    border: 3px solid #41ADAE;
    border-radius: 10px;
  }

  .pocket-container {
    display: flex;
  }

  // 各リスト
  .pose-container {
    border: 3px solid #41ADAE;
    border-radius: 10px;
  }
</style>
