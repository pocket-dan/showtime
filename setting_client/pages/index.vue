<template>
  <v-container
    fluid
    grid-list-md
  >
    <v-layout
      row
    >
      <v-flex
        v-for="pose in poses"
        :key="pose.id"
        :class="{ pocket: pose.id === 7 }"
       >
        <Pose
          :pose="pose"
          :cards="cardsByPose(pose.id)"
          @set-cards="setCards($event)"
          @remove-pose="removePose($event)"
          :class="{ pocket_box: pose.id === 7 }"
        />
      </v-flex>
      <v-btn @click="postRelation()" class="ma-2" color="teal decide-pose-action" dark large>
        決定<v-icon dark right>check_circle</v-icon>
      </v-btn>
      <v-btn @click="resetRelation($event)" class="ma-2" color="blue-grey reset-pose-action" dark large>
        リセット<v-icon dark left>remove_circle</v-icon>
      </v-btn>
    </v-layout>
  </v-container>
</template>

<script>
  import { mapState } from 'vuex'
  import Pose from '@/components/Pose'

  export default {
    components: {
      Pose
    },
    computed: {
      ...mapState({ poses: state => state.poses.data }),
    },
    methods: {
      cardsByPose(poseId) {
        return this.$store.getters['cards/cardsByPose'](poseId)
      },
      removePose(payload) {
        this.$store.dispatch('poses/remove', payload)
      },
      setCards(payload) {
        console.log(payload)
        this.$store.dispatch('cards/set', payload)
      },
      postRelation() {
        this.$axios.post('/post_data', this.$store.state.cards)
      },
      resetRelation() {
        location.reload()
      }
    }
  }
</script>

<style lang="scss" scoped>
  .container {
    width: 100%;
  }

  .pocket {
    width: 100%;
    position: fixed;
    bottom: 20px;
    left: 0;
  }
  .pocket_box {
    overflow: scroll;
    height: 100%;
  }

  .decide-pose-action {
    position: fixed;
    bottom: 170px;
    left: 174px;
  }

  .reset-pose-action {
    position: fixed;
    left: 5px;
    bottom: 170px;
  }
</style>
